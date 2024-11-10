import os
from utils import dice_loss
import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image
import config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weight_path = config.weight_path

data_path = config.train_data

save_path = config.save_path

val_path = config.val_path
#val_path =config.val_data

learning_rate = config.inital_learning_rate



load_model = config.weight_path

experiment_file = config.experiment_name
train_file = config.train_name

experiment_path = os.path.join("./train_results",experiment_file)
train_path = os.path.join(experiment_path,train_file)
param_path = os.path.join(config.save_param,experiment_file)
param_file = os.path.join(param_path,train_file)


if __name__ == '__main__':

    mkdir(experiment_path)
    mkdir(train_path)
    mkdir(param_path)
    mkdir(param_file)
    txtepoch1 = open(os.path.join(train_path,"US.txt"),'w')
    txtepoch2 = open(os.path.join(train_path, "other.txt"),'w')
    txtval = open(os.path.join(train_path, "val.txt"), 'w')

    num_classes = 2  # +1是背景也为一类
    data_loader = DataLoader(MyDataset(data_path), batch_size=config.train_batch_size, shuffle=True)
    data_loader2 = DataLoader(MyDataset(config.train_data_2), batch_size=config.train_batch_size, shuffle=True)

    net = UNet(num_classes).to(device)
    if load_model:
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！\n')
    else:
        print('not successful load weight\n')

    opt = optim.Adam(net.parameters(),lr=learning_rate)

    loss_fun = nn.CrossEntropyLoss()

    scheduler_s = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100, 125, 150, 175, 200], gamma=0.5)

    epoch = 1
    while epoch < config.num_epoch+1:

        epoch_loss=[]
        epoch_dice=[]
        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader)):

            image, segment_image = image.to(device), segment_image.to(device)
            net.train()
            out_image = net(image)

            train_loss = loss_fun(out_image, segment_image.long()) + dice_loss(F.softmax(out_image, dim=1).float(),F.one_hot(segment_image.long(),num_classes).permute(0, 3, 1, 2).float(), multiclass=True)
            train_loss *= config.value

            opt.zero_grad()
            train_loss.backward()
            opt.step()
            epoch_loss.append(train_loss.item())

            _image = image[0]
            _segment_image = torch.unsqueeze(segment_image[0], 0) * 255
            _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255

            img = torch.stack([_segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')


            epoch_dice.append(dice_coeff(_out_image/255,_segment_image/255).item())

            if i % 100== 0:
                print('US======>>epoch:{},item:{},loss:{},dice:{}'.format(epoch,i,train_loss.item(),dice_coeff(_out_image/255,_segment_image/255).item()))
                #txtitem.write('epoch:{},item:{},loss:{},dice:{}\n'.format(epoch,i,train_loss.item(),dice_coeff(_out_image/255,_segment_image/255).item()))
                #txtitem.write('\r\n')


        print(f"US======>>epoch:{epoch} avarge_loss:{np.mean(epoch_loss)} avarge_dice:{np.mean(epoch_dice)}")
        txtepoch1.write("epoch:{} avarge_loss:{} avarge_dice:{}\n".format(epoch,np.mean(epoch_loss),np.mean(epoch_dice)))
        #txtepoch.write('\r\n')
        ###############################################

        for i, (image, segment_image) in enumerate(tqdm.tqdm(data_loader2)):
            image, segment_image = image.to(device), segment_image.to(device)
            net.train()
            out_image = net(image)

            train_loss = loss_fun(out_image, segment_image.long()) + dice_loss(F.softmax(out_image, dim=1).float(),
                                                                               F.one_hot(segment_image.long(),
                                                                                         num_classes).permute(0, 3, 1,
                                                                                                              2).float(),
                                                                               multiclass=True)


            opt.zero_grad()
            train_loss.backward()
            opt.step()
            epoch_loss.append(train_loss.item())

            _image = image[0]
            _segment_image = torch.unsqueeze(segment_image[0], 0) * 255
            _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255

            img = torch.stack([_segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{i}.png')
            epoch_dice.append(dice_coeff(_out_image / 255, _segment_image / 255).item())

            if i % 100== 0:
                print('other======>>epoch:{},item:{},loss:{},dice:{}'.format(epoch,i,train_loss.item(),dice_coeff(_out_image/255,_segment_image/255).item()))
                #txtitem.write('epoch:{},item:{},loss:{},dice:{}\n'.format(epoch,i,train_loss.item(),dice_coeff(_out_image/255,_segment_image/255).item()))
                #txtitem.write('\r\n')


        print(f"other======>>epoch:{epoch} avarge_loss:{np.mean(epoch_loss)} avarge_dice:{np.mean(epoch_dice)}")
        txtepoch2.write("epoch:{} avarge_loss:{} avarge_dice:{}\n".format(epoch,np.mean(epoch_loss),np.mean(epoch_dice)))





        ################################################
        scheduler_s.step()

        if epoch % 1 == 0:

            print("-------------------val_start-------------------")

            epoch_acc = 0

            for filename in os.listdir(os.path.join(val_path,"Labels")):

                in_image = Image.open(os.path.join(val_path,"Images")+"/"+filename).convert("RGB")

                in_image = np.expand_dims(np.transpose(preprocess_input(np.array(in_image, np.float32)), (2, 0, 1)), 0)

                with torch.no_grad():

                    in_image = torch.from_numpy(in_image)

                    in_image = in_image.cuda()
                    net.eval()

                    pred = net(in_image)

                    seg = torch.argmax(pred[0], dim=0).unsqueeze(0)*255

                    save_image(seg/255,"./val_data/pred/{}".format(filename))

            val_dice = read_dice("val_data/pred", "val_data/labels")
            print("avarge_val_acc:{}".format(val_dice))
            txtval.write("avarge_val_acc:{}".format(val_dice))
            txtval.write("\r\n")

            print("-------------------val_end-------------------")
            torch.save(net.state_dict(),os.path.join(param_file,"unet_model-ep-{}.pth".format(epoch)))
            print('save successfully!\n')

        epoch += 1



