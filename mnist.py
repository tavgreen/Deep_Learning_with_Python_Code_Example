import torch #import torch library
import torch.nn as nn #import torch neural network library
import torch.nn.functional as F #import functional neural network module
import torch.optim as optim #import optimizer neural network module
from torch.autograd import Variable #import variable that connect to automatic differentiation
from torchvision import datasets, transforms #import torchvision for datasets and transform

class Model(nn.Module): #class model
	def __init__(self):
		super(Model, self).__init__() #load super class for training data
		self.conv1 = nn.Conv2d(1, 10, 5) #Convolutional modul: input, output, kernel
		self.conv2 = nn.Conv2d(10, 20, 5) #Convolutional modul: input, output, kernel
		self.maxpool = nn.MaxPool2d(2) #maxpooling modul: kernel
		self.relu = nn.ReLU() #activation relu modul
		self.dropout2d = nn.Dropout2d() #dropout modul
		self.dropout = nn.Dropout() #dropout modul
		self.fc1 = nn.Linear(320, 50) #Fully Connected modul: input, output
		self.fc2 = nn.Linear(50, 10)# Fully Connected modul: input, output

	def forward(self, x): #feed forward
		layer1 = self.relu(self.maxpool(self.conv1(x))) # layer1 = x -> conv1 -> maxpool -> relu
		layer2 = self.relu(self.maxpool(self.dropout2d(self.conv2(layer1)))) #layer2 = layer1 -> conv2 -> dropout -> maxpool -> relu
		layer3 = layer2.view(-1, 320) #make it flat from 0 - 320
		layer4 = self.relu(self.fc1(layer3)) #layer4 = layer3 -> fc1 -> relu
		layer5 = self.fc2(layer4) #layer5 = layer4 -> fc2
		return F.log_softmax(layer5) #softmax activation to layer5
#load train and test loader that will be normalized and shuffle
train_loader = torch.utils.data.DataLoader( 
	datasets.MNIST('dataset/',train=True, download=False, 
		 transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
		 batch_size=1000, shuffle=True)
test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('dataset/',train=False, 
		 transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),
		 batch_size=1000, shuffle=True)

model = Model() #load graph / model
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9) #load optimizer SGD with momentum 0.9 and learning rate 0.01
for epoch in range(10): # train epoch = 10
	model.train() #training
	for batch_idx, (data, label) in enumerate(train_loader): #enumerate train_loader per batch-> index, (data, label) ex: 0, (img1, 4)... 1, (img2, 2)
		data, label = Variable(data), Variable(label) #create torch variable and enter each data and label into it
		optimizer.zero_grad()
		output = model(data) #enter data into model, save in output
		train_loss = F.nll_loss(output, label) #nll = negative log likehood loss between output and label. it useful for classification problem with n class
		train_loss.backward() #compute gradient
		optimizer.step() #update weight
		if batch_idx % 10 == 0: #display step
			print('Train Epochs: {}, Loss: {:.6f} '.format(epoch, train_loss.data[0] )) #print
	model.eval() #evaluation/testing
	test_loss = 0
	correct = 0
	for data, label in test_loader: #separate data and label
		data, label = Variable(data,volatile=True), Variable(label) #create torch variable and enter data and label into it
		output = model(data) #enter data into model, save in output
		test_loss += F.nll_loss(output, label, size_average=False).data[0] #
		pred = output.data.max(1, keepdim=True)[1] #prediction result
		correct += pred.eq(label.data.view_as(pred)).cpu().sum() #if label=pred then correct++
	test_loss /= len(test_loader.dataset) #compute test loss
	print('\nAverage Loss: {:.4f}, Accuracy: {:.0f}'.format(test_loss,  100. * correct / len(test_loader.dataset)))