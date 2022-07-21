close all; clear; clc;
cd

data_dir = '../data/';
sr = 0;                % pixel size of half random point
data_num = 1;
% Nxy = 128;  Nz = 7;  dz = 1.2e-3; 
% Nxy = 128;  Nz = 25;  dz = 0                                                                                                                                                                                                                                 .725e-3;
Nxy = 128;  Nz = 32;  dz = 1e-4;
%Nxy = 64;  Nz = 32;  dz = 0.725e-3;     
holoDataType = 1;
%%
lambda = 660e-9;     % Illumination wavelength
pps    = 20e-6;      % pixel pitch of CCD camera
z0     = 5e-3;       % Distance between the hologram and the center plane of the 3D object
z_range = z0 + (0:Nz-1)*dz;   % axial depth span of the object

NA = pps*Nxy/2/z0;
delta_x = lambda/(NA)
delta_z = 2*lambda/(NA^2)

%% set params 
params.lambda = lambda;
params.pps = pps;
params.z = z_range;
params.Ny = Nxy;
params.Nx = Nxy;
ori_otf3d = ProjKernel(params);

params.K                =  2;                % Spatial  Overasampling Factor
params.T                =  30;  % Temporal Overasampling Factor
params.Qmax             =  2;               % Maximum Threshold


noise_level = 500;   % DB of the noise
switch holoDataType
    case 1
        if Nz==3
            ppv_min = 1e-3;
            ppv_max = 5e-3;            
        elseif Nz==7
            ppv_min = 1e-3;
            ppv_max = 5e-3;
        elseif Nz==15
            ppv_min = 5e-4;
            ppv_max = 25e-4;
        elseif Nz==32
            ppv_min = 2e-4;
            ppv_max = 1e-3;
        end

%         data_dir = [data_dir,'Nz', num2str(Nz), '_Nxy', num2str(Nxy),'_kt',num2str(params.T),'_ks',num2str(params.K)];
        
%     case 2
%         Test data_single with varying ppvs
%         ppvs = [1 2 3 4 5 6 7 8 9 10]*1e-3;
%         group_num = length(ppvs);
%         
%         data_num = 100;         % number of test data_single
         
%         data_dir = [data_dir, 'test_ppv_Nz', num2str(Nz),'_dz', num2str(dz*1e6),'um'];
    case 3
        % Fixed ppv
            ppv_min = 2e-4;
            ppv_max = 2e-4;

end

if ppv_min == ppv_max
    ppv_text = [num2str(ppv_min,'%.e')];
else
    ppv_text = [num2str(ppv_min,'%.e') '~' num2str(ppv_max,'%.e')];
end

data_dir = [data_dir,'Nz', num2str(Nz), '_Nxy', num2str(Nxy),'_kt',num2str(params.T),'_ks',num2str(params.K),'_ppv',ppv_text];
        
if not(exist(data_dir,'dir'))
    mkdir(data_dir)
end

 
%% generate training data 
for idx = 1:data_num
    data = zeros(Nxy,Nxy);
    label = zeros(Nz,Nxy,Nxy);
    y = zeros(params.T,Nxy*params.K,Nxy*params.K);
    N_random = randi([round(ppv_min*Nxy*Nxy*Nz) round(ppv_max*Nxy*Nxy*Nz)], 1, 1); % particle concentration
    obj = randomScatter(Nxy, Nz, sr, N_random);   % randomly located particles
    %imagesc(plotdatacube(obj)); title(['3D object with particle of ' num2str(N_random) ]); axis image; drawnow; colormap(hot);
    t_o = (1-obj);
    [data_single] = gaborHolo(t_o, ori_otf3d, noise_level);
    otf3d = permute(ori_otf3d,[3,1,2]);% just permute for saving 
    data(:,:) = data_single;
    data = (data-min(min(data)))/(max(max(data))-min(min(data)));
    label(:,:,:) = permute(obj,[3,1,2]);% [Nz,Nxy,Nxy]
    y(:,:,:) = generateQIS(params,data);
%     save([data_dir,'/',num2str(idx),'.mat'],'data','label','otf3d','y');
    disp(idx)
end


AT = @(plane) (BackwardProjection(plane, ori_otf3d));
figure;
imagesc(plotdatacube(permute(label,[2,3,1]))); title('Last object'); axis image; drawnow; colormap(hot); colorbar; axis off;
figure; imagesc(data(:,:)); title('Hologram'); axis image; drawnow; colormap(gray); colorbar; axis off;
temp = abs(real(AT(data)));  %temp = (temp- min(temp(:)))./(max(temp(:))-min(temp(:)));
figure; imagesc(plotdatacube(temp)); title('Gabor reconstruction'); axis image; drawnow; colormap(hot); colorbar; axis off;


% reconstruct DH by benchmark algorithm 
QmapLR  =  ones([Nxy,Nxy]);
alpha = params.K^2*(params.Qmax-1);
IM_qs   =  imageReconstruct(permute(y,[2,3,1]),params.K,alpha,QmapLR);
figure; imagesc(IM_qs(:,:)); title('Hologram_qis'); axis image; drawnow; colormap(gray); colorbar; axis off;
