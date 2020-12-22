import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from utils.siCluster_utils import *
from utils.parameters import *
import glob
import shutil
import copy
import csv
from sklearn.metrics import silhouette_score



def extract_inhabited_cluster(args):
    convnet = models.resnet18(pretrained=True)
    convnet = torch.nn.DataParallel(convnet)    
    ckpt = torch.load('./checkpoint/ckpt_vanilla_cluster_nk_z14_100.t7')
    convnet.load_state_dict(ckpt, strict = False)
    convnet.module.fc = nn.Sequential()
    convnet.cuda()
    cluster_transform =transforms.Compose([
                      transforms.Resize(256),
                      transforms.CenterCrop(224),
                      transforms.RandomGrayscale(p=1.0),
                      transforms.ToTensor(),
                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])    
    
    clusterset = GPSDataset('./meta_data/meta_inhabited_16_19_nk_z14.csv', './data/NK', cluster_transform)
    clusterloader = torch.utils.data.DataLoader(clusterset, batch_size=128, shuffle=False, num_workers=1)
    
    print("#######################rural#########################")
    for i in range(20,3,-1):
        deepcluster = Kmeans(i, 'rural')
        features = compute_features(clusterloader, convnet, len(clusterset), 128) 
        print(features.shape)
        clustering_loss, p_label = deepcluster.cluster(features[968:])
        score = silhouette_score(features[968:], p_label.detach().cpu().numpy(), metric="euclidean")
        print("score of cluster {} in rural is {}".format(i, score))
        deepcluster = Kmeans(i, 'city')
        features = compute_features(clusterloader, convnet, len(clusterset), 128) 
        clustering_loss, p_label = deepcluster.cluster(features[:968])
        score = silhouette_score(features[:968], p_label.detach().cpu().numpy(), metric="euclidean")
        print("score of cluster {} in city is {}".format(i, score))
    labels = p_label.tolist()
    f = open('./meta_data/meta_inhabited_16_19_nk_z14.csv', 'r', encoding='utf-8')
    images = []
    rdr = csv.reader(f)
    for line in rdr:
        images.append(line[0])
    f.close()
    images.pop(0)    
    rural_cluster = []
    for i in range(0, len(images)):
        rural_cluster.append([images[i], labels[i]])
    return rural_cluster

def extract_nature_cluster(args):
    f = open('./meta_data/meta_nature_16_19_nk_z14.csv', 'r', encoding='utf-8')
    images = []
    rdr = csv.reader(f)
    for line in rdr:
        images.append(line[0])
    f.close()
    images.pop(0)    
    nature_cluster = []
    cnum = args.rural_cnum
    for i in range(0, len(images)):
        nature_cluster.append([images[i], cnum])
        
    return nature_cluster



def main(args):
    # make cluster directory
    inhabited_cluster = extract_inhabited_cluster(args)
    nature_cluster = extract_nature_cluster(args)
    total_cluster = inhabited_cluster + nature_cluster
    cnum = args.rural_cnum
    cluster_dir = './data/{}/'.format(args.cluster_dir)
    if not os.path.exists(cluster_dir):
        os.makedirs(cluster_dir)
        for i in range(0, cnum + 1):
            os.makedirs(cluster_dir + str(i))
    else:
        raise ValueError
    
    for img_info in total_cluster:
        cur_dir = './data/NK/' + img_info[0]
    
        new_dir = cluster_dir + str(img_info[1])
        new_file = cluster_dir + str(img_info[1])+'/'+img_info[0][-14:-4]+'_'+img_info[0].split('/')[0]+'.png'

        shutil.copy(cur_dir, new_dir)
        os.rename(cluster_dir + str(img_info[1])+'/'+img_info[0][-14:], new_file)

    file_list = glob.glob("./{}/*/*.png".format(args.cluster_dir))
    grid_dir = cluster_dir + args.grid
    f = open(grid_dir, 'w', encoding='utf-8')
    wr = csv.writer(f)
    wr.writerow(['y_x', 'cluster_id'])
    
    for file in file_list:
        file_split = file.split("/")
        folder_name = file_split[2]
        file_name = file_split[-1].split(".")[0]
        wr.writerow([file_name, folder_name])
    f.close()
    

if __name__ == "__main__":
    args = extract_cluster_parser()
    main(args)    
    