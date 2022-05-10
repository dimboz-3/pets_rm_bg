
from PIL import Image

import streamlit as st
import numpy as np

import torch
import torchvision
from torchvision import transforms
from resnet_unet import ResNetUNet 
import os
import requests
from io import BytesIO
import gdown

@st.cache()
def load_model():
	model = ResNetUNet(1)
	
	path = 'resnet_unet_rm_bg_for_pets.pth'
	gdown.download(id = "1-4eVIIfim8vf_kMm9uPMj_JVeOxdbYV2")
	assert(os.path.exists(path))

	model.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
	model.eval()
	return model


def predict(
		img: Image.Image,
		model,
		) -> Image.Image:
		
	trans = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
		])
	#attention mask
	img_tensor = trans(img)[None,...]
	with torch.set_grad_enabled(False):
		att_mask = model.forward(img_tensor)[0].detach()
	att_mask = torch.sigmoid(att_mask)
	att_mask = att_mask.numpy().transpose((1, 2, 0))
	return att_mask[:,:,0]


def load_var_from_session(key):
	if key in st.session_state:
		return  st.session_state[key]
	else:
		return None
#smaller img if needed
def fit_img(img):
	width, height = img.size
	max_w_h = max(width, height)
	max_len = 960
	ratio = max_len/max_w_h
	if max(width, height) > max_len:
		img = img.resize( (int(width*ratio), int(height*ratio)), Image.BILINEAR)
	return img

if __name__ == '__main__':

	model = load_model()

	st.title('Welcome To Project find Pets on photo!')

	img 			= load_var_from_session("img")
	file_old 		= load_var_from_session("file_old")
	file_url_old 	= load_var_from_session("file_url_old")
	att_mask 		= load_var_from_session("att_mask")

	NEW_IMG = False

	#upload file
	file = st.file_uploader('Upload An Image')
	if file and (file_old != file):
		try:
			img = Image.open(file)
			file_old = file
			NEW_IMG = True
		except:
			 st.write("Uploaded file is not image!!!")

	#load by url	
	file_url = st.text_input('image url')
	if file_url and (file_url_old != file_url):
		try:
			response = requests.get(file_url)
			img = Image.open(BytesIO(response.content))	
			file_url_old = file_url
			NEW_IMG = True
		except:
			 st.write("can't load image from that url")


	if NEW_IMG:

		img=fit_img(img)
		img=np.array(img)

	if img is not None:
		st.title("Here is the image you've selected")
		st.image(img)

	if NEW_IMG:
		##calculate model prediction
		att_mask = predict(img,model)

		st.session_state["att_mask"]=att_mask


	if img is not None:

		st.title("Attention mask")
		st.image(Image.fromarray(np.uint8(att_mask*255)))
		
		att_threshold = st.slider("attention threshold",
			min_value=0.0,
			max_value=1.0,
			value=0.3,
			step=0.01)
		
		st.title("sharp boundaries")
		cut_img_sharp = np.where(att_mask[:,:,None]>att_threshold,img,255)
		st.image(cut_img_sharp)
		
		st.title("smooth boundaries")
		cut_img_smooth = np.where(att_mask[:,:,None]>att_threshold,
			img*att_mask[:,:,None]+255*(1-att_mask[:,:,None]), 255)
		st.image(Image.fromarray(np.uint8(cut_img_smooth)))
		
	st.session_state["img"]=img
	st.session_state["file_old"]=file_old
	st.session_state["file_url_old"]=file_url_old

		