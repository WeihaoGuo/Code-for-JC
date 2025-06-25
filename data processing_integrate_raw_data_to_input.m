clear;clc;
%% 
%load G:\WarmPoolFlux\Data_ICOADS\all\时间质量控制\ws_icoads_3std.mat N;ws=N;clear N;
load G:\WarmPoolFlux\Data_ICOADS\ws_icoads_3std_new.mat N;ws=N;clear N;
ws=reshape(ws,126,66,12*120);
%%
% load G:\WarmPoolFlux\Data_ICOADS\all\时间质量控制\sst_icoads_3std.mat N;sst=N;clear N;
% sst=reshape(sst,126,66,12*120);
load G:\WarmPoolFlux\Data_ICOADS\sst_icoads_3std_new.mat N;ws=N;clear N;
sst=reshape(pred1,126,66,12*120);clear pred1

%%
load G:\WarmPoolFlux\Data_ICOADS\ta_icoads_3std_new.mat N;ws=N;clear N;
ta=reshape(ta,126,66,12*120);
%%
load G:\WarmPoolFlux\Data_ICOADS\cld_icoads_3std_new.mat N;ws=N;clear N;
cld=reshape(cld,126,66,12*120);
%%
load G:\WarmPoolFlux\Data_ICOADS\slp_icoads_3std_new.mat N;ws=N;clear N;
slp=reshape(slp,126,66,12*120);
%%
load G:\WarmPoolFlux\Data_ICOADS\qa_icoads_3std_new.mat N;ws=N;clear N;
qa=reshape(qa,126,66,12*120);
%%
ws_rss=ncread('ws_v07r01_198801_202212.nc3.nc','wind_speed');
ws_rss=ws_rss(:,:,25:384);ws_rss(ws_rss<0)=nan;
lon=0.5:359.5;lat=-89.5:89.5;
%% 确定画图区域
lonmax=157.5;
lonmin=32.5;
latmax=47.5;
latmin=-17.5;
iny=find(lat>=latmin&lat<=latmax);
y=lat(iny);
inx=find(lon>=lonmin&lon<=lonmax);
ws_rss=ws_rss(inx,iny,:); clear l* i*
ws_rss1(:,:,1081:1440)=ws_rss;
ws_rss1(:,:,1:1080)=nan;
ws_rss=ws_rss1;clear ws_rss1
%% 地理信息
lon=32.5:1:157.5;
lat=-17.5:1:47.5;
x=zeros(126*66,120,12)*NaN;
y=zeros(126*66,120,12)*NaN;
for i=1:length(lat)
    for j=1:length(lon)
        x(length(lon)*(i-1)+j,:,:)=lon(j);
        y(length(lon)*(i-1)+j,:,:)=lat(i);
    end
end
clear i j lon lat
x=permute(x,[1,3,2]);y=permute(y,[1,3,2]);
%% 时间信息
year=zeros(126*66,120,12)*NaN;
month=zeros(126*66,120,12)*NaN;
for i=1900:2019
    year(:,i-1899,:)=i;
end
clear i
for i=1:12
    month(:,:,i)=i;
end
clear i
year=permute(year,[1,3,2]);month=permute(month,[1,3,2]);
x=reshape(x,[],1);
y=reshape(y,[],1);
month=reshape(month,[],1);
year=reshape(year,[],1);
ws=reshape(ws,[],1);
sst=reshape(sst,[],1);
ta=reshape(ta,[],1);
slp=reshape(slp,[],1);
cld=reshape(cld,[],1);
qa=reshape(qa,[],1);

ws_rss=reshape(ws_rss,[],1);

%%
input=[ws,qa,sst,ta,cld,slp,x,y,year,month,ws_rss];
save('input_data.mat');