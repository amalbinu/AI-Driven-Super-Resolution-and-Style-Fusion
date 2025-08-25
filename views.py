from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from .models import *
import os
import numpy as np
import cv2
from django.conf import settings
from tensorflow.keras.models import load_model
from django.core.files.storage import FileSystemStorage
from django.views.decorators.csrf import csrf_exempt
import matplotlib.pyplot as plt
from io import BytesIO
import base64

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
from torch.optim import LBFGS
import os
from .vgg19 import Vgg19

IMAGENET_MEAN_255 = [123.675, 116.28, 103.53]
IMAGENET_STD_NEUTRAL = [1, 1, 1]

def load_image(img_path,target_shape="None"):
    
    if not os.path.exists(img_path):
        raise Exception(f'Path not found: {img_path}')
    img = cv.imread(img_path)[:, :, ::-1]                   # convert BGR to RGB when reading
    if target_shape is not None:
        if isinstance(target_shape, int) and target_shape != -1:
            current_height, current_width = img.shape[:2]
            new_height = target_shape
            new_width = int(current_width * (new_height / current_height))
            img = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_CUBIC)
        else:
            img = cv.resize(img, (target_shape[1], target_shape[0]), interpolation=cv.INTER_CUBIC)
    img = img.astype(np.float32)
    img /= 255.0
    return img

def prepare_img(img_path, target_shape, device):
    
    img = load_image(img_path, target_shape=target_shape)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        transforms.Normalize(mean=IMAGENET_MEAN_255, std=IMAGENET_STD_NEUTRAL)])
    img = transform(img).to(device).unsqueeze(0)
    return img

def save_image(img, img_path):
    if len(img.shape) == 2:
        img = np.stack((img,) * 3, axis=-1)
    cv.imwrite(img_path, img[:, :, ::-1])                   # convert RGB to BGR while writing

def generate_out_img_name(config):
    
    prefix = os.path.basename(config['content_img_name']).split('.')[0] + '_' + os.path.basename(config['style_img_name']).split('.')[0]
    suffix = f'{config["img_format"][1]}'
    return prefix + suffix

def save_and_maybe_display(optimizing_img, dump_path, config, img_id, num_of_iterations):
    
    saving_freq = -1
    out_img = optimizing_img.squeeze(axis=0).to('cpu').detach().numpy()
    out_img = np.moveaxis(out_img, 0, 2)

    if img_id == num_of_iterations-1 :
        img_format = config['img_format']
        out_img_name = str(img_id).zfill(img_format[0]) + img_format[1] if saving_freq != -1 else generate_out_img_name(config)
        dump_img = np.copy(out_img)
        dump_img += np.array(IMAGENET_MEAN_255).reshape((1, 1, 3))
        dump_img = np.clip(dump_img, 0, 255).astype('uint8')
        cv.imwrite(os.path.join(dump_path, out_img_name), dump_img[:, :, ::-1])
    

def prepare_model(device):
    
    model = Vgg19(requires_grad=False, show_progress=True)
    content_feature_maps_index = model.content_feature_maps_index
    style_feature_maps_indices = model.style_feature_maps_indices
    layer_names = model.layer_names
    content_fms_index_name = (content_feature_maps_index, layer_names[content_feature_maps_index])
    style_fms_indices_names = (style_feature_maps_indices, layer_names)
    return model.to(device).eval(), content_fms_index_name, style_fms_indices_names

def gram_matrix(x, should_normalize=True):
    
    (b, ch, h, w) = x.size()
    features = x.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t)
    if should_normalize:
        gram /= ch * h * w
    return gram

def total_variation(y):
    
    return torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

def build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    
    target_content_representation = target_representations[0]
    target_style_representation = target_representations[1]
    current_set_of_feature_maps = neural_net(optimizing_img)
    current_content_representation = current_set_of_feature_maps[content_feature_maps_index].squeeze(axis=0)
    content_loss = torch.nn.MSELoss(reduction='mean')(target_content_representation, current_content_representation)
    style_loss = 0.0
    current_style_representation = [gram_matrix(x) for cnt, x in enumerate(current_set_of_feature_maps) if cnt in style_feature_maps_indices]
    for gram_gt, gram_hat in zip(target_style_representation, current_style_representation):
        style_loss += torch.nn.MSELoss(reduction='sum')(gram_gt[0], gram_hat[0])
    style_loss /= len(target_style_representation)
    tv_loss = total_variation(optimizing_img)
    total_loss = config['content_weight'] * content_loss + config['style_weight'] * style_loss + config['tv_weight'] * tv_loss
    return total_loss, content_loss, style_loss, tv_loss

def make_tuning_step(neural_net, optimizer, target_representations, content_feature_maps_index, style_feature_maps_indices, config):
    
    def tuning_step(optimizing_img):
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index, style_feature_maps_indices, config)
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return total_loss, content_loss, style_loss, tv_loss
    return tuning_step

def neural_style_transfer(config):
    
    content_img_path = os.path.join(config['content_images_dir'], config['content_img_name'])
    style_img_path = os.path.join(config['style_images_dir'], config['style_img_name'])
    out_dir_name = 'combined_' + os.path.split(content_img_path)[1].split('.')[0] + '_' + os.path.split(style_img_path)[1].split('.')[0]
    dump_path = os.path.join(config['output_img_dir'], out_dir_name)
    os.makedirs(dump_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    content_img = prepare_img(content_img_path, config['height'], device)
    style_img = prepare_img(style_img_path, config['height'], device)
    
    init_img = content_img
    
    optimizing_img = Variable(init_img, requires_grad=True)
    neural_net, content_feature_maps_index_name, style_feature_maps_indices_names = prepare_model(device)
    print(f'Using VGG19 in the optimization procedure.')
    content_img_set_of_feature_maps = neural_net(content_img)
    style_img_set_of_feature_maps = neural_net(style_img)
    target_content_representation = content_img_set_of_feature_maps[content_feature_maps_index_name[0]].squeeze(axis=0)
    target_style_representation = [gram_matrix(x) for cnt, x in enumerate(style_img_set_of_feature_maps) if cnt in style_feature_maps_indices_names[0]]
    target_representations = [target_content_representation, target_style_representation]
    num_of_iterations = 100
    
    optimizer = LBFGS((optimizing_img,), max_iter=num_of_iterations, line_search_fn='strong_wolfe')
    cnt = 0

    def closure():
        nonlocal cnt
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        total_loss, content_loss, style_loss, tv_loss = build_loss(neural_net, optimizing_img, target_representations, content_feature_maps_index_name[0], style_feature_maps_indices_names[0], config)
        if total_loss.requires_grad:
            total_loss.backward()
        with torch.no_grad():
            print(f'L-BFGS | iteration: {cnt:03}, total loss={total_loss.item():12.4f}, content_loss={config["content_weight"] * content_loss.item():12.4f}, style loss={config["style_weight"] * style_loss.item():12.4f}, tv loss={config["tv_weight"] * tv_loss.item():12.4f}')
            save_and_maybe_display(optimizing_img, dump_path, config, cnt, num_of_iterations)
        cnt += 1
        return total_loss
    optimizer.step(closure)
    return dump_path

# Load the model
# model = load_model('high_app/static/model/model.pt')
model = torch.load('high_app/static/model/model.pt', map_location="cpu")

def index_page(request):
    return render(request, "index.html")

def about_page(request):
    return render(request, "about.html")

def service_page(request):
    return render(request, "service.html")

def why_page(request):
    return render(request, "why.html")
    
def login_page(request):
    return render(request, "login.html")

def home_page(request):
    return render(request, "home.html")

def high_page(request):
    return render(request, "high_image.html")

def image_page(request):
    return render(request, "style_transfer.html")




def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def convert_images(request):
    if request.method == 'POST' and request.FILES.get('image_file'):
        # Save the uploaded file
        uploaded_file = request.FILES['image_file']
        fs = FileSystemStorage()
        file_path = fs.save(uploaded_file.name, uploaded_file)
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
        # Preprocess the uploaded image
        test_img = preprocess_image(full_path)

        # Predict the high-res image
        predicted_img = model.predict(test_img)
        predicted_img = np.squeeze(predicted_img)
        predicted_img = np.clip(predicted_img * 255.0, 0, 255).astype(np.uint8)

        # Save the predicted image to the media directory
        predicted_file_path = os.path.join(settings.MEDIA_ROOT, f"enhanced_{uploaded_file.name}")
        cv2.imwrite(predicted_file_path, cv2.cvtColor(predicted_img, cv2.COLOR_RGB2BGR))

        # Encode the predicted image to display on the front-end
        _, buffer = cv2.imencode('.png', cv2.cvtColor(predicted_img, cv2.COLOR_RGB2BGR))
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        # Render the response
        return render(request, 'high_image.html', {
            'uploaded_image_url': fs.url(file_path),
            'enhanced_image': encoded_image
        })

    return render(request, 'high_image.html')

def process_style_transfer(content_image_path, style_image_path):
    # Replace this with your image processing logic
    result_image_path = "path_to_generated_image/result.jpg"
    return result_image_path

@csrf_exempt
def style_transfer(request):
    if request.method == "POST" and request.FILES:
        content_image = request.FILES.get("content_image")
        style_image = request.FILES.get("style_image")
        
        fs = FileSystemStorage(location="media/uploads/")
        content_image_path = fs.save(content_image.name, content_image)
        style_image_path = fs.save(style_image.name, style_image)

        content_image_full_path = fs.path(content_image_path)
        style_image_full_path = fs.path(style_image_path)

        # Call the processing function
        result_image_path = process_style_transfer(content_image_full_path, style_image_full_path)

        # Respond with the result image URL
        result_image_url = fs.url(result_image_path)
        return JsonResponse({"result_image_url": result_image_url})
    
    return JsonResponse({"error": "Invalid request"}, status=400)

def convert_image(request):
    if request.method == 'POST' and request.FILES['image_file']:
        image_file = request.FILES['image_file'] 

        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)


        return HttpResponse("<script>alert('Image Enhanced Successfully');window.location.href='/high_page/'</script>")

    return redirect('high_page')

def style_transfer(request):
    if request.method == 'POST' and 'content_image' in request.FILES and 'style_image' in request.FILES:
        content_image = request.FILES['content_image']
        style_image = request.FILES['style_image']

        fs = FileSystemStorage()
        content_filename = fs.save(content_image.name, content_image)
        style_filename = fs.save(style_image.name, style_image)


        return HttpResponse("<script>alert('Images Uploaded Successfully');window.location.href='/image_page/'</script>")

    return redirect('image_page')


def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        password_confirmation = request.POST.get('password_confirmation')
        mobile = request.POST.get('mobile')

        if not username or not email or not password or not password_confirmation or not mobile:
            return HttpResponse("<script>alert('All fields are required');window.location.href='/login_page/';</script>")

        if password != password_confirmation:
            return HttpResponse("<script>alert('Passwords do not match');window.location.href='/login_page/';</script>")

        if Users.objects.filter(username=username).exists():
            return HttpResponse("<script>alert('User with this username already exists');window.location.href='/login_page/';</script>")

        if Users.objects.filter(email=email).exists():
            return HttpResponse("<script>alert('Email is already registered');window.location.href='/login_page/';</script>")

        user = Users(username=username, email=email, password=password, mobile=mobile)
        user.save()


        return HttpResponse("<script>alert('Registration Successful!');window.location.href='/login_page/';</script>")

    return render(request, 'login.html')


def login(request):
    if request.method == 'POST':
 
        email = request.POST.get('email')
        password = request.POST.get('password')
        try:
            user = Users.objects.get(email=email)
        except Users.DoesNotExist:
            return HttpResponse("<script>alert('Email not found! Please check your credentials.');window.location.href='/login/';</script>")

        user = Users.objects.get(email=email, password= password)

        if user is not None:
            request.session['email'] = user.email
            request.session['user_id'] = user.r_id
            return HttpResponse("<script>alert('User Login Sucessfull');window.location.href='/home_page/';</script>")
        else:
            return HttpResponse("<script>alert('Invalid password! Please check your credentials.');window.location.href='/login/';</script>")

    return render(request, 'login.html')


def logout(request):
    request.session.flush() 
    return render(request, "index.html")

