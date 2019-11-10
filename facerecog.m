clear all;
close all;

load Dataset; %load the training set we work on

M = 6; %number of dataset images
N = 512; %new size of the dataset iamges

%%plotting the training set
%To_imshow = [ st.data{1}  st.data{2} st.data{3} ; st.data{4} st.data{5} st.data{6}];
%figure
%imshow(To_imshow,'Initialmagnification','fit')
%title('Data set')
%% compute mean

average_image = zeros(N);
for i = 1 : M
    st.data{i} = im2single(st.data{i});
    average_image = average_image + st.data{i};
end
average_image = average_image/M;
% %plotting the average image
% figure
% imshow(average_image,'Initialmagnification','fit')
% title('The dataset average')

%% normalize (remove mean)

for i = 1 : M
    st.dataAvg{i}  = st.data{i} - average_image;
end
%%plotting the training set subtracted of the mean
%To_imshow  = [ st.dataAvg{1}  st.dataAvg{2} st.dataAvg{3}; st.dataAvg{4} st.dataAvg{5} st.dataAvg{6}];
%figure
%imshow(To_imshow,'Initialmagnification','fit')
%title('Dataset minus the average')


%% generate A = [ img1(:)  img2(:) ...  imgM(:) ]

A = zeros(N*N,M); % N^2*M
for i=1:M
    A(:,i) = st.dataAvg{i}(:);
end
% Covariance matrix in small dimension (transposed)
C = A'*A; % M*M
%%plotting the covariance matrix
%figure
%imagesc(C)
%title('Covariance')

%% eigen vectros  in small dimension

[Veigvec,Deigval] = eig(C);% V(M*M), D(M*M) diagonal with M eigen values
% the eigen face in large dimension  A*Veigvec is eigen vector of Clarge
Vlarge = A*Veigvec; % [N^2*M] * [M*M] = N^2*M
% reshape each column (N^2*1) to obtain each eigen face (N*N)
eigenfaces = cell(1,M);
for i=1:M
    eigenfaces{i} = reshape(Vlarge(:,i),N,N);
end
lambdas = diag(Deigval); %obtaining eigen values
[lambdas_sorted,lambdas_sorted_index]=sort(lambdas,'descend'); %sorting the eigen values in descending order
% %plotting the eigenfaces (ghost faces) in descending order
% To_imshow  = [ eigenfaces{lambdas_sorted_index(1)}  eigenfaces{lambdas_sorted_index(2)} eigenfaces{lambdas_sorted_index(3)}; eigenfaces{lambdas_sorted_index(4)} eigenfaces{lambdas_sorted_index(5)} eigenfaces{lambdas_sorted_index(6)} ];
% figure
% imshow(To_imshow,'Initialmagnification','fit')
% title('eigenfaces')

%% weights

sel = M; % select number of eigen faces (max = M)
wi = zeros(sel,M);
for mi=1:M  % image number
    for i=1:sel   % eigen face for coeff number
        % we get the weights matrix (sel*M) by multiplying
        % the transposed eigenfaces (1*N^2) by the corresponding images (N^2*1) 
        wi(mi,i) = eigenfaces{lambdas_sorted_index(i)}(:)' * A(:,mi) ;
    end
end

%% classify a new image

%first the user enters the name of the test image
testFace_name = input('Enter the test image name: ');
testFace = imread(testFace_name);
testFace = rgb2gray(testFace);
testFace = imresize(testFace,[N N]);
testFace = im2single(testFace);

%figure
%imshow(testFace,'Initialmagnification','fit')
%title('Test face')

%then we normalize the test face
testFaceColumn = testFace(:)-average_image(:);

%then we calculate the new weights of the new image
wface = zeros(1,sel);
for j = 1 : sel
    wface(j)  =  eigenfaces{lambdas_sorted_index(j)}(:)' * testFaceColumn ;
end


% compute distance
diffWeights = zeros(1,M);
for mi = 1 : M
    sumdist=0;
    for j=1:sel
        sumdist = sumdist + (wface(j) -wi(mi,j)).^2;
    end
    diffWeights(mi) = sqrt(sumdist);
end

% defining a threshold to the face classes
% test image has to be frontal, bare headed, average brightness and white background
% (i.e. passort image).
theta = 2.75e+3;
if min(diffWeights) > theta
    fprintf('The image you have entered is of an UNKNOWN person\n');
else
    index = find(diffWeights == min(diffWeights));
    person_detected_name = st.names{index};
    fprintf('The person is %s \n',person_detected_name);
    person_detected_image = st.data{index};
    To_imshow = [testFace person_detected_image];
    figure
    imshow(To_imshow,'Initialmagnification','fit')
    title('Testface ------------------------------------- Matched image')
end
