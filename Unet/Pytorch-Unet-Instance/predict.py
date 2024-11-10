from data import *
from net import *
from torchvision.utils import save_image
import config

model_pth = config.prd_model
test_name = config.experiment_name+"_"+config.test_name
save_path1 = os.path.join("./test_results",test_name)
mkdir(save_path1)
# save_path2 = os.path.join("./test_results",test_name+"_跨域")
# mkdir(save_path2)
txt=open(os.path.join(save_path1,(config.train_name+"_predict_dice.txt")),'w')

def test(pred_data,save_path,pred_net):
    for filename in os.listdir(os.path.join(pred_data,"Images")):
        i_image = Image.open(os.path.join(pred_data, "Images") + "/" + filename).convert("RGB")
        i_image = np.expand_dims(np.transpose(preprocess_input(np.array(i_image, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            pred_net.eval()
            i_image = torch.from_numpy(i_image)
            i_image = i_image.cuda()
            out_image = pred_net(i_image)
            o_image = torch.argmax(out_image[0], dim=0).unsqueeze(0)*255
            save_image(o_image/255,os.path.join(save_path,"{}".format(filename)))
    dice=[]
    for filename in os.listdir(save_path):
        if not filename.endswith(".txt"):
            img = Image.open(save_path + "/" + filename).convert("RGB")
            target = Image.open(os.path.join(os.path.join(pred_data, "Labels"),filename) )

            output = threshold_demo(np.array(img))
            output.save(os.path.join(save_path, "{}".format(filename)))
            target = transform(target)
            output = transform(output)
            final_dice = dice_coeff(output, target)
            dice.append(final_dice.item())
            print("{},dice:{}".format(filename, final_dice.item()))
            txt.write("{},dice:{}\n".format(filename, final_dice.item()))
            #txt.write('\r\n')


    return np.mean(dice),np.std(dice)

if __name__ == '__main__':
    pred_net = UNet(num_classes=2).cuda()
    pred_net.load_state_dict(torch.load(model_pth))

    a,b=test(config.test_data,save_path1,pred_net)
    print(f"dice_average:{a},标准差:{b}")
    txt.write(f"dice_average:{a},std:{b}")





















