#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
from pathlib import Path
import glob
from numpy.lib.function_base import extract
import torch
import random
import argparse
import numpy as np

sys.path.append(os.getcwd())
from models import utils
import pickle
from utils.build_vocab import Vocabulary

from torch.utils.tensorboard import SummaryWriter
from models.stage1_model import * 
from torch.utils.data import DataLoader, dataset
from datasets.stage1_dataset import Stage1DataSet,collate_fn
from schedule import WarmupCosineSchedule

def train(configs):
    
    ###############################
    #
    #参数加载
    #
    ###############################
    train_args = configs["train_args"]
    model_args = configs["model_args"]
    optimizer_args = configs["optimizer_args"]
    scheduler_args = configs["scheduler_args"]
    device = torch.device(train_args["device"])
    vocabulary =  pickle.load(open(configs["vocabulary"], "rb"))
    train_dataset = Stage1DataSet(vocabulary,configs["trainjsonfile"],**train_args)
    train_loader = DataLoader(train_dataset, batch_size=train_args["batch_size"], shuffle=True, num_workers=train_args["num_workers"], drop_last=True,collate_fn=collate_fn) 
    
    ###############################
    #
    #logging加载
    #
    ###############################
    os.makedirs(configs["checkpoint_dir"],exist_ok=True)
    os.makedirs(configs["log_dir"],exist_ok=True)
    logger = utils.genlogger(os.path.join(configs["log_dir"], "train.log"))
    logger.info("Storing files in: {}".format(configs["checkpoint_dir"]))
    utils.pprint_dict(configs, logger.info)
    tb_writer = SummaryWriter(Path(configs["log_dir"]) / "run")
    
    ###############################
    #
    #网络、loss、优化器声明
    #
    ################################
    embedder_net = Stage1Encoder(len(vocabulary),**model_args)
    if "pretrained_word_embedding" in configs:
        embeddings = np.load(configs["pretrained_word_embedding"])
        embedder_net.load_word_embeddings(embeddings, tune=configs["tune_word_embedding"], projection=True)        
    embedder_net = embedder_net.to(device)
    ge2e_loss = GE2ELoss(device)
    optimizer = torch.optim.SGD([
                    {'params': embedder_net.parameters()},
                    {'params': ge2e_loss.parameters()}
                ], lr=optimizer_args["lr"])
    scheduler = WarmupCosineSchedule(optimizer,configs["scheduler_args"]["warmup_steps"],configs["epochs"])
    ###############################
    #训练
    #################################
    embedder_net.train()
    iteration = 0
    eer = 100
    for e in range(configs["epochs"]):
        total_loss = 0
        print("----------------------------Begin training-----------------------------------------------")
        logger.info("----------------------------Begin training-----------------------------------------------")
        for batch_id, batch in enumerate(train_loader): 
            embedder_net.to(device).train()
            captions,lens,id= batch
            captions = captions.to(device)
            perm = random.sample(range(0, len(lens)), len(lens))
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
            captions = captions[perm]
            lens = lens[perm]
            
            #gradient accumulates
            optimizer.zero_grad()
            embeddings = embedder_net(captions,lens)["caption_embeds"]
            embeddings = embeddings[unperm]
            lens = lens[unperm]
            embeddings = torch.reshape(embeddings, (train_args["batch_size"], train_args["uttnumbers"], embeddings.size(1)))
            
            #get loss, call backward, step optimizer
            loss = ge2e_loss(embeddings) #wants (Speaker, Utterances, embedding)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder_net.parameters(), 3.0)
            torch.nn.utils.clip_grad_norm_(ge2e_loss.parameters(), 1.0)
            
            optimizer.step()
            tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], iteration)

            total_loss = total_loss + loss
            tb_writer.add_scalar("loss/train", loss.item(), iteration)
            iteration += 1
            if (batch_id + 1) % optimizer_args["log_interval"] == 0:
                mesg = "{0}\tEpoch:{1}[{2}/{3}],Iteration:{4}\tLoss:{5:.4f}\tTLoss:{6:.4f}\t\n".format(time.ctime(), e+1,
                        batch_id+1, len(train_dataset)//train_args["batch_size"], iteration,loss, total_loss / (batch_id + 1))
                print(mesg)
                logger.info(mesg)
            # avg_loss = total_loss / (batch_id + 1)
        mesg = "-----------------------------Epoch:[{0}/{1}] is finished!-----------------------------\n".format(e+1,configs["epochs"])
        print(mesg)
        logger.info(mesg)
        temp = test(configs,embedder_net).cpu().data.numpy()
        tb_writer.add_scalar("eer/train", temp, iteration)
        logger.info(str(temp))
        if temp < eer:
            eer = temp
            print(temp)
            
            embedder_net.eval().cpu()
            ckpt_model_filename = "best.pth"
            ckpt_model_path = os.path.join(configs["checkpoint_dir"], ckpt_model_filename)
            key  = e+1
            torch.save(embedder_net.state_dict(), ckpt_model_path)
            embedder_net.to(device).train()
        scheduler.step()
    
    
    logger.info("The best model is in epoch "+str(key))

def test(configs,embedder_net):
    
    test_args = configs["test_args"]
    model_args = configs["model_args"]
    device = torch.device(test_args["device"])
    vocabulary = pickle.load(open(configs["vocabulary"], "rb"))

    test_dataset = Stage1DataSet(vocabulary,configs["testjsonfile"],**test_args)
    test_loader = DataLoader(test_dataset, batch_size=test_args["batch_size"], shuffle=True, num_workers=test_args["num_workers"], drop_last=True,collate_fn=collate_fn)


    embedder_net.eval()
    embedder_net = embedder_net.to(device)
    
    avg_EER = 0
    for e in range(test_args['epoch']):

        batch_avg_EER = 0
        for batch_id, batch in enumerate(test_loader):
            captions,lens,id = batch
            captions = captions.to(device)  
            captions = torch.reshape(captions,(test_args['batch_size'],test_args["uttnumbers"],captions.shape[-1]))
            lens = torch.reshape(lens,(test_args['batch_size'],test_args["uttnumbers"]))
            verification_batch,enrollment_batch = torch.chunk(captions, 2, dim=1)
            verification_lens,enrollment_lens = torch.chunk(lens, 2, dim=1)

            enrollment_batch = torch.reshape(enrollment_batch, (-1, enrollment_batch.size(2)))
            verification_batch = torch.reshape(verification_batch, (-1, verification_batch.size(2)))
            enrollment_lens = torch.reshape(enrollment_lens, (-1,))
            verification_lens = torch.reshape(verification_lens, (-1,))


            perm = random.sample(range(0,verification_batch.size(0)), verification_batch.size(0))
            unperm = list(perm)
            for i,j in enumerate(perm):
                unperm[j] = i
                
            verification_batch = verification_batch[perm]
            verification_lens = verification_lens[perm]
            enrollment_embeddings = embedder_net(enrollment_batch,enrollment_lens)["caption_embeds"]
            verification_embeddings = embedder_net(verification_batch,verification_lens)["caption_embeds"]
            verification_embeddings = verification_embeddings[unperm]
            verification_lens = verification_lens[unperm]

            enrollment_embeddings = torch.reshape(enrollment_embeddings, (test_args['batch_size'], -1, enrollment_embeddings.size(1)))
            verification_embeddings = torch.reshape(verification_embeddings, (test_args['batch_size'], -1, verification_embeddings.size(1)))
            
            enrollment_centroids = get_centroids(enrollment_embeddings)
            
            sim_matrix = get_cossim(verification_embeddings, enrollment_centroids)
            
            # calculating EER
            diff = 1; EER=0; EER_thresh = 0; EER_FAR=0; EER_FRR=0
            
            for thres in [0.01*i for i in range(100)]:
                sim_matrix_thresh = sim_matrix>thres
                
                FAR = (sum([sim_matrix_thresh[i].float().sum()-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(test_args['batch_size']))])
                /(test_args['batch_size']-1.0)/(float(verification_embeddings.shape[1]))/test_args['batch_size'])
    
                FRR = (sum([verification_embeddings.shape[1]-sim_matrix_thresh[i,:,i].float().sum() for i in range(int(test_args['batch_size']))])
                /(float(verification_embeddings.shape[1]))/test_args['batch_size'])
                
                # Save threshold when FAR = FRR (=EER)
                if diff> abs(FAR-FRR):
                    diff = abs(FAR-FRR)
                    EER = (FAR+FRR)/2
                    EER_thresh = thres
                    EER_FAR = FAR
                    EER_FRR = FRR
            batch_avg_EER += EER
        avg_EER += batch_avg_EER/(batch_id+1)
            
    avg_EER = avg_EER / test_args['epoch']
    return avg_EER
        
def extract_emdedding(configs):
    
    embedding_args = configs["embedding_args"]
    model_args = configs["model_args"]
    device = torch.device(embedding_args["device"])
    vocabulary = pickle.load(open(configs["vocabulary"], "rb"))
    embedding_dataset = Stage1DataSet(vocabulary,configs["extractembeddingjsonfile"],embedding= True,**embedding_args)
    embedding_loader = DataLoader(embedding_dataset, batch_size=embedding_args["batch_size"], shuffle=False, num_workers=embedding_args["num_workers"], drop_last=True,collate_fn=collate_fn)    

    embedder_net = Stage1Encoder(len(vocabulary),**model_args)
    if "pretrained_word_embedding" in configs:
        embeddings = np.load(configs["pretrained_word_embedding"])
        embedder_net.load_word_embeddings(embeddings, tune=configs["tune_word_embedding"], projection=True)
    ckpt_model_path = os.path.join(configs["checkpoint_dir"],"best.pth")
    embedder_net.load_state_dict(torch.load(ckpt_model_path))
    embedder_net.eval()
    embedder_net = embedder_net.to(device)
    
    logger = utils.genlogger(os.path.join(configs["log_dir"],"extract_emdedding.log"))
    os.makedirs(configs["embedding_path"],exist_ok=True)
    logger.info("Storing caption embedding files in: {}".format(configs["embedding_path"]))
    print("Storing caption embedding files in: {}".format(configs["embedding_path"]))
    
    for batch_id, batch in enumerate(embedding_loader):
        captions,lens,audioid = batch
        captions = captions.to(device)  
        captions = torch.reshape(captions,(embedding_args['batch_size'],embedding_args["uttnumbers"],captions.shape[-1]))
        lens = torch.reshape(lens,(embedding_args['batch_size'],embedding_args["uttnumbers"]))
        captions = torch.reshape(captions, (-1, captions.size(2)))
        lens = torch.reshape(lens, (-1,))

        embeddings = embedder_net(captions,lens)["caption_embeds"].squeeze().data.cpu().numpy()
        audioiddir = os.path.join(configs["embedding_path"],audioid[0])
        os.makedirs(audioiddir,exist_ok=True)
        logger.info("the captions embedding of {} is doned~~~~~~".format(audioid[0]))
        for i in range(len(embeddings)):
            np.save(os.path.join(audioiddir,str(i)+".npy"),embeddings[i])
        np.save(os.path.join(audioiddir,"caption.npy"),np.average(embeddings,0))
     
    logger.info("~~~~All of {} is finished~~~~~~".format(len(embedding_dataset)))
    
if __name__=="__main__":
    # parser = argparse.ArgumentParser()
    istrain = sys.argv[1]
    jsonpath = sys.argv[2]
    configs = utils.parse_config_or_kwargs(jsonpath)
    if istrain == "train":
        train(configs)
    else: 
        extract_emdedding(configs)