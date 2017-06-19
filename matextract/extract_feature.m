% ------------------------------------------------------------------------
% caffe/matlab/+caffe/imagenet/ilsvrc_2012_mean.mat contains mean_data that
% is already in W x H x C with BGR channels
clear all

initialization
caffe.set_mode_gpu;caffe.set_device(0);% Set GPU ID
split=1;
%%
%------------------------------------------------%

img_path='/dir-to-input-images/';
file=dir([img_path,'/*.jpg']);
mean_path='/dir-to-mean-file/';
mean_data = caffe.io.read_mean([mean_path,'mean-file-name.binaryproto']);
mean_data=single(mean_data);



width=224;height=224;%Set input image size
batch_size=16;
input_data=zeros(width,height,3,batch_size,'single');
feature=[];
Num=length(file);
for i=1:Num

	img_name=file(i).name;
	im=imread([img_path,'/',img_name]);

% Convert an image returned by Matlab's imread to im_data in caffe's data
% format: W x H x C with BGR channels
	im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
	im_data = permute(im_data, [2, 1, 3]);  % flip width and height
	im_data = single(im_data);  % convert from uint8 to single
	im_data = imresize(im_data, [width,height], 'bilinear');  % resize im_data
	im_data = im_data - mean_data;  % subtract mean_data (already in W x H x C, BGR)
	input_data(:,:,:,mod(i-1,batch_size)+1)=im_data;
	input_blob={input_data};
	if(mod(i,batch_size)==0||i==Num)
        net.forward(input_blob);
        tmp=net.blobs('pool5').get_data(); % Set feature blob
        tmp0=reshape(tmp,2048,batch_size); % Set feature dim
        feature=[feature,tmp0];

	end

	if (mod(i,50)==0)
	fprintf(['processing ',num2str(i), ' imgs\n']);
	end
end
feat=feature(:,1:Num);
save('feature-file-name.mat','feat')
caffe.reset_all
