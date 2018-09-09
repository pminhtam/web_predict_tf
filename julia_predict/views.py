from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import os
import cv2
# Create your views here.
from ML_model.utils import Julia
def predict(request):
    if request.method == 'POST':
        file = request.FILES['img']
        # print(file.mane)
        # print(file)
        # print(type(file))
        # print(file.seek(0))
        if file:
            if os.path.isfile("media/img.png"):
                os.remove("media/img.png")
            fs = FileSystemStorage()
            filename = fs.save("img.png", file)  # tên file là thời gian hiện tại + đuôi file
            img = cv2.imread("media/img.png")
            # print(img)
            model = Julia.getInstance()
            print(model)
            result = model.predict(img)
        return render(request, 'julia_index.html',{'result':result})
    return render(request,'julia_index.html')