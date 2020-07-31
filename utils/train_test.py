import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torchvision
import torchsummary
from torchsummary import summary
import torch


from tqdm import tqdm


def train(model, device, train_loader, optimizer):
	train_losses = []
	train_acc = []

	model.train()
	pbar = tqdm(train_loader)
	correct = 0
	processed = 0
	criterion=nn.NLLLoss().to(device)
	
	for batch_idx, (data, target) in enumerate(pbar):
		# get samples
		data, target = data.to(device), target.to(device)

		# Init
		optimizer.zero_grad()
		# In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
		# Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

		# Predict
		y_pred = model(data)

		# Calculate loss
		
		loss  = criterion(y_pred, target)
		
		train_losses.append(loss)

		#Backpropagation
		loss.backward()
		optimizer.step()

		# Update pbar-tqdm
		
		pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
		correct += pred.eq(target.view_as(pred)).sum().item()
		processed += len(data)

		pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
	train_acc.append(100*correct/processed)
	return train_acc[-1]
	

def test(model, device, test_loader):
	test_losses_l1 = []
	test_acc_l1 = []
	model.eval()
	test_loss = 0
	correct = 0
	processed = 0
	
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			

			test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
			processed += len(data)

	test_loss /= len(test_loader.dataset)
	test_losses_l1.append(test_loss)

	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
		test_loss, correct, processed,
		100. * correct / processed))
	
	test_acc_l1.append(100. * correct / processed)
	return test_acc_l1[-1]

def fit_generator(model, device, train_loader, test_loader, optimizer, scheduler, start_epoch, num_epoch, plot_acc = False):
	train_acc = []
	test_acc = []
	for epoch in range(start_epoch, start_epoch+num_epoch):
		curr_lr=optimizer.param_groups[0]['lr']
		print(f'Epoch: {epoch} Learning_Rate {curr_lr}')
		train_acc1 = train(model, device, train_loader, optimizer)
		test_acc1 = test(model, device, test_loader)
		#print('Test accuracy:', test_acc1)
		train_acc.append(train_acc1)
		test_acc.append(test_acc1)
		if "ReduceLROnPlateau" in  str(type(scheduler)):
			scheduler.step(test_acc1)
		elif "OneCycleLR" in  str(type(scheduler)):
			scheduler.step()
	
	if plot_acc:
		plt.plot(range(start_epoch, start_epoch+num_epoch), train_acc, label= 'Train Accuracy')
		plt.plot(range(start_epoch, start_epoch+num_epoch), test_acc, label= 'Test Accuracy')
		plt.legend()
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy")
		plt.title("Epoch wise train and test accuracies")
	
	
def predict(model, device, test_loader):
	pred_all=[]
	model.eval()

	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)

			pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
			#incorrect_pred.append(pred.eq(target.view_as(pred)))
			pred_all +=list(pred.squeeze().cpu().numpy())

	return pred_all

def get_misclassified(pred,labels):
	misclassified = []
	correct = []
	for i in (range(len(pred))):
		if pred[i] != labels[i] : misclassified.append((i,pred[i],labels[i]))
		else : correct.append((i,pred[i],labels[i]))
	return correct,	misclassified

	
def print_misclassified(model, device, test_loader, n, labels_list):
	'''
	n : number of misclassified Images per class
	labels_list : List of label names
	'''
	model.to(device)
	fail=[]
	for eval_data, eval_target in test_loader:
		eval_data, eval_target = eval_data.to(device), eval_target.to(device)
		eval_out = model(eval_data)
		target_lbl=eval_target.cpu().numpy()
		pred_lbl=eval_out.argmax(1).cpu().numpy()
		#print(pred_lbl)
		#print(eval_target)
		for i in range(test_loader.batch_size):
			if target_lbl[i] != pred_lbl[i]:fail.append([eval_data[i].cpu().numpy(),target_lbl[i],pred_lbl[i]])
		if len(fail) > n * 20: break
	
	#List (Class-wise)
	fail_n = []
	for i in range(len(labels_list)):
		fail_i = [x for x in fail if x[1] == i]
		if len(fail_i) > n : fail_i = fail_i[:10]
		fail_n.append(fail_i)
	
	for i in range(len(fail_n)):
		miscl = len(fail_n[i])
		fig = plt.figure(figsize=(40, 40 * miscl ))
		for j in range(miscl):
			ax=fig.add_subplot(1, miscl, j+1)
			img=np.transpose(fail_n[i][j][0], (1, 2, 0))
			min,max = img.min(), img.max()
			img = (img - min)/(max-min)
			ax.imshow(img)
			ax.axis('off')
			gs1 = gridspec.GridSpec(1, miscl)
			gs1.update(wspace=0.025, hspace=0.05)
			ax.set_title("Actual : "+labels_list[fail_n[i][j][1]]+"\n  Predicted : "+labels_list[fail_n[i][j][2]])
		plt.savefig('Misclassified_{}.jpg'.format(i), orientation = 'landscape', bbox_inches = 'tight')
		plt.show()
