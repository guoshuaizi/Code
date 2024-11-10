for i1 = 159:1:200
Files1 = dir(fullfile('E:\标签处理CTspacing\label_spacing')); % 读取文件夹内的mat格式的文件

name1=Files1(i1+3).name;           %读取struct变量的格式
folder1=Files1(i1+3).folder;


%Files8 = dir(fullfile('D:\data')); % 读取文件夹内的mat格式的文件
Files8 = dir(fullfile('E:\标签处理CTspacing\label_spacing\*.nii.gz')); % 读取文件夹内的mat格式的文件


name8=Files8(i1+1).name;           %读取struct变量的格式
folder8=Files8(i1+1).folder;

%nii8 = load_nii( [folder8,'\',name8,'\segmentation.nii.gz']);  % 装载.nii数据
nii8 = load_nii( [folder8,'\',name8]);  % 装载.nii数据
img8 = nii8.img;  %文件包含img和head，img是图像数据


nii1 = load_nii( [folder1,'\',name1,]);  % 装载.nii数据
img1 = nii1.img;  %文件包含img和head，img是图像数据

J = imbinarize(img1);%3D二值化

s = regionprops(J,'BoundingBox');  %可能不止一个连通区域，所以结果可能不止一个
ss = regionprops(J,'Centroid');
b = s.BoundingBox;
a = s.BoundingBox;

a(1) = a(1) - a(4)*0.25;   %改外接矩形大小
if (a(1)<1)
    a(1)=1;
end
a(4) = a(4)*1.5;
if ((a(4)+a(1))>nii8.hdr.dime.dim(2))
    a(4)=nii8.hdr.dime.dim(2)-a(1)-1;
end
a(2) = a(2) - a(5)*0.4;   %改外接矩形大小
if (a(2)<1)
    a(2)=1;
end
a(5) = a(5)*1.8;
if ((a(5)+a(2))>nii8.hdr.dime.dim(3))
    a(5)=nii8.hdr.dime.dim(3)-a(2)-1;
end
a(3) = a(3) - a(6)/4;   %改外接矩形大小
if (a(3)<1)
    a(3)=1;
end
a(6) = a(6)*3/2;
if ((a(6)+a(3))>nii8.hdr.dime.dim(4))
    a(6)=nii8.hdr.dime.dim(4)-a(3)-1;
end

J = imcrop3(img8,a);

J = abs(J);
nii = make_nii(J); % 形成nii格式的数据

ind = strfind(name1,'.gz');

save_nii(nii,strcat('E:\zhende_matlab_project\result2\',name1(1:ind-5),'.nii'));
gzip('E:\zhende_matlab_project\result2\*.nii') %将所有nii形成各自的nii.gz

Files2 = dir(fullfile('E:\标签处理CTspacing\data2_spacing\*.nii.gz')); % 读取文件夹内的mat格式的文件

name2=Files2(i1+1).name;           %读取struct变量的格式
folder2=Files2(i1+1).folder;

nii2 = load_nii( [folder2,'\',name2]);  % 装载.nii数据
JJ = nii2.img;  %文件包含img和head，img是图像数据

JJ = imcrop3(JJ,a);

% JJ = abs(JJ);
nii3 = make_nii(JJ); % 形成nii格式的数据

ind = strfind(name2,'.gz');

save_nii(nii3,strcat('E:\zhende_matlab_project\result2_picture\',name2(1:ind-5),'.nii'));
gzip('E:\zhende_matlab_project\result2_picture\*.nii') %将所有nii形成各自的nii.gz
end