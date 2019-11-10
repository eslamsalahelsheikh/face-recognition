clear all;
close all;

N = 512 ; % new image size

%abubakr
abubakr = image_input('a1.JPG',N);
% hamdy
hamdy = image_input('b1.JPG',N);
% eslam
eslam = image_input('c1.JPG',N);
% weley
welely = image_input('d1.JPG',N);
% osama
osama = image_input('e1.JPG',N);
% wizo
wizo = image_input('f1.JPG',N);

%% store
st.names = {'abubakr','hamdy','eslam','welely','osama','wizo'};
st.data{1} = abubakr;
st.data{2} = hamdy;
st.data{3} = eslam;
st.data{4} = welely;
st.data{5} = osama;
st.data{6} = wizo;

data  = [abubakr hamdy eslam; welely osama wizo];
figure
imshow(data,'Initialmagnification','fit');title('Data set')

save Dataset st;