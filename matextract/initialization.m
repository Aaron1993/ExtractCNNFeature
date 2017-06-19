if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end

model_dir = '/dir-to-deploy-prototxt/';
net_model = [model_dir 'deploy.prototxt'];
weight_dir = '/dir-to-trained-model/';
net_weights = [weight_dir,'final.caffemodel'];
phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Could not find caffemodel files.');
end
if ~exist(net_model, 'file')
  error('Could not find deploy prototxt.');
end
% Initialize a network
net = caffe.Net(net_model, net_weights, phase);
