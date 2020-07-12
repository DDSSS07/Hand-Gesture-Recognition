from PIL import Image

def resize(image):
    width = 100
    img = Image.open(image)
    wpercent = (width/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((width,hsize), Image.ANTIALIAS)
    img.save(image)

for img in range(0,100):
    
    resize("Dataset/PalmTest/palm_" + str(img) + '.png')


