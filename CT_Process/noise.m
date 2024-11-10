% 路径
SamplePath1 =  'D:/bqgc/bai/';  %存储图像的路�?
savepath = 'D:/bqgc/output/';        %
fileExt = '*.png';  %待读取图像的后缀�?
%获取�?��路径
files = dir(fullfile(SamplePath1,fileExt)); 
len1 = size(files,1);
%遍历路径下每�?��图像
for i=1:len1
   fileName = strcat(SamplePath1,files(i).name); 
   image = imread(fileName);
   
   
   image = 1 - image;
   
   BB = bwconncomp(image);
   
   ab = 1;
   while ((BB.NumObjects>1) && (ab == 1))             %仅保留面积最大的唯一�?��
   numPixels = cellfun(@numel,BB.PixelIdxList);
   [biggest,idx] = min(numPixels);
        if biggest < 2000
            image(BB.PixelIdxList{idx}) = 0;
        else
            ab = 0;
        end
   BB = bwconncomp(image);
   end
   
   image = 1 - image;
   
   fileName2 = strcat(savepath,files(i).name);
   imwrite(image, fileName2);
   
   
end