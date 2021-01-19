clc;clear;

fpara = [0.2 0.5 0.3];
infolder = 'test_images/';
outfolder = 'predicted_outputs/';

Gfolder = [outfolder 'fov180_salmap'];   % fov360
Mfolder = [outfolder 'fov120_Msalmap'];  % fov120
Lfolder = [outfolder 'fov90_Lsalmap'];   % fov90
Cfolder = [outfolder 'Csalmap'];         % combined saliency map from 3 fovs

mkdir(Gfolder);
mkdir(Mfolder);
mkdir(Lfolder);
mkdir(Cfolder);

fprintf('Predicting Global Saliency Maps\n');
cmd = ['python ', '03-predict_fov360.py ' infolder ' ' Gfolder];
status = system(cmd);
if status ~= 0
    fprintf('There is something wrong with SALGAN. Please check and run again.\n');
    %exit(-1);
end

fileName = ['im360.jpg']
im360 = imread([infolder fileName]);    
fprintf('Read image.\n');
[iml0 imw0 c] = size(im360);
im360 = imresize(im360, [1024 2048]);
headmove_h = (0:10:80);
headmove_v = (0:10:80);
vfov90 = 90;
vfov120 = 120;
iml = size(im360,1);
imw = size(im360,2);
pad = 0;
fprintf('Making multiple cube projection...\n');
mkdir(['MCP90']);
mkdir(['MCP120']);
for hh = 1:length(headmove_h)
    offset=round(headmove_h(hh)/360*imw);
    im_turned = [im360(:,imw-offset+1:imw,:) im360(:,1:imw-offset,:)];
    for hv = 1:length(headmove_v)
        [out90] = equi2cubic(im_turned, iml, vfov90, headmove_v(hv));
        [out120] = equi2cubic(im_turned, iml, vfov120, headmove_v(hv));
        for i=1:6
            imwrite(cell2mat(out90(i)), ['MCP90\' num2str(hv) '_' num2str(hh) '_im360_', num2str(i), '.jpg']);
            imwrite(cell2mat(out120(i)), ['MCP120\' num2str(hv) '_' num2str(hh) '_im360_', num2str(i), '.jpg']);
        end
    end      
end
clear out;


% Salgan
fprintf('Getting saliency from SALGAN model...\n');
fov120_pred_vp = [outfolder 'fov120_pred_vp'];
mkdir(fov120_pred_vp);
cmd = ['python ', '03-predict_fov120.py'  ' '  'MCP120'  ' ' fov120_pred_vp];
status = system(cmd);
if status ~= 0
    fprintf('There is something wrong with SALGAN. Please check and run again.\n');
    exit(-1);
end


fprintf('Getting saliency from SALGAN model...\n');
fov90_pred_vp = [outfolder 'fov90_pred_vp'];
cmd = ['python ', '03-predict_fov90.py'  ' '  'MCP90'  ' ' fov90_pred_vp];
status = system(cmd);
if status ~= 0
    fprintf('There is something wrong with SALGAN. Please check and run again.\n');
    exit(-1);
end



fprintf('Fusing saliency maps for FoV120...\n');
fov120 = zeros(1024, 1024, 3, 6);
im_salgan_0 = zeros(1024, 1024*2, 1);
    

for v = 1:length(headmove_v)  
    for h=1:length(headmove_h)
    for i=1:6
        filename = saliencyList(i+(v-1)*6*length(headmove_h)+(h-1)*6).name
        cubsal = double(imresize(imread([outfolder 'predi_salmap/' filename]), [1026, 1026]));
        fov120(:,:,1,i) = cubsal(217:217+591, 217:217+591);
        fov120(:,:,2,i) = cubsal(217:217+591, 217:217+591);
        fov120(:,:,3,i) = cubsal(217:217+591, 217:217+591);
        
    end    
[hv, rest] = strtok(filename(6:end),'_');
hv = str2double(rest(2));
hh = str2double(rest(3));
im_salgan = cubic2equi(0, fov120(:,:,:,5), fov120(:,:,:,6), fov120(:,:,:,4), fov120(:,:,:,2), fov120(:,:,:,1), fov120(:,:,:,3));
out = equi2cubic(im_salgan, 1024, 90, -headmove_v(hv));

im_salgan = cubic2equi(-headmove_h(hh),cell2mat(out(5)),cell2mat(out(6)),cell2mat(out(4)),cell2mat(out(2)),cell2mat(out(1)),cell2mat(out(3)));
im_salgan = im_salgan(:,:,1);
im_salgan = double(im_salgan) + im_salgan_0;
im_salgan_0 = im_salgan;
    end
end

im_salgan = im_salgan./(h*v);
im_salgan = im_salgan./max(im_salgan(:)*255);
im_salgan120 = imresize(im_salgan, [1024 2048]);



fprintf('Fusing saliency maps for FoV90...\n');
saliencyList = dir(fov90_pred_vp);
im_cub_sal = zeros(iml, iml, 6, 3);
im_salgan_0 = zeros(iml,iml*2);

for v = 1:length(headmove_v)
    for h=1:length(headmove_h)

    for i=1:6
        filename = saliencyList(i+(v-1)*6*length(headmove_v)+(h-1)*6+2).name
        filefolder = saliencyList(i+(v-1)*6*length(headmove_v)+(h-1)*6+2).folder;
        cubsal = double(imread([filefolder '/', filename]));
        cubsal = imresize(cubsal, [iml iml]);
        im_cub_sal(:,:,i,1) = cubsal;
        im_cub_sal(:,:,i,2) = cubsal;
        im_cub_sal(:,:,i,3) = cubsal;
    end    
[hv, rest] = strtok(filename,'_');
hv = str2double(hv);
hh = str2double(strtok(rest,'_'));
im_salgan = cubic2equi(0,im_cub_sal(:,:,5,:), im_cub_sal(:,:,6,:), im_cub_sal(:,:,4,:), im_cub_sal(:,:,2,:), im_cub_sal(:,:,1,:), im_cub_sal(:,:,3,:));
out = equi2cubic(im_salgan, iml, vfov120, -headmove_v(hv));
im_salgan = cubic2equi(-headmove_h(hh),cell2mat(out(5)),cell2mat(out(6)),cell2mat(out(4)),cell2mat(out(2)),cell2mat(out(1)),cell2mat(out(3)));
im_salgan = im_salgan(:,:,1);
im_salgan = double(im_salgan)+im_salgan_0;
im_salgan_0 = im_salgan;
    end
end
im_salgan = im_salgan./(h*v);
im_salgan = im_salgan./max(im_salgan(:)*255);
im_salgan90 = imresize(im_salgan, [1024 2048]);



fprintf('Fusing 3 FoVs saliency maps...\n');

Gsalmap = double(imread([Gfolder '/' fileName]));
Gsalmap = Gsalmap./max(Gsalmap(:)*255);
Gsalmap = imresize(Gsalmap, [1024 2048]);

Msalmap = im_salgan120;
Msalmap = Msalmap./max(Msalmap(:)).*255;
imwrite(uint8(Msalmap), [Mfolder '/' fileName]);  

Lsalmap = im_salgan90;
Lsalmap = Lsalmap./max(Lsalmap(:)).*255;
imwrite(uint8(Lsalmap), [Lfolder '/' fileName]);  

im_combined = fpara(1).*Lsalmap + fpara(2).* Msalmap + fpara(3).* Gsalmap;
im_combined = im_combined./max(im_combined(:))*255;
imwrite(uint8(im_combined), [Cfolder 'predi_im360_2048x1024.jpg']);    

fprintf('Done!');
end
