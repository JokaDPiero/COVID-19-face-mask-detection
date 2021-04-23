import cv2
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input

# Capture the Video from webcam..
cap = cv2.VideoCapture(0)

# load haarcascade file
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

#Loading the pre-trained model to predict mask or no mask
model = load_model("vgg19_face_model_keras.h5")



results={
    0:'Without Mask',
    1:'Mask'
}

color={
    0:(0,0,255),#red
    1:(0,255,0) #green
}

# Infinte Loop
while True:

	# Read the Webcam Image
    ret, frame  = cap.read()
    
    frame=cv2.flip(frame,1) #(not mirror image)
    
    # If not able to read image
    if ret == False:
        continue


	# Detect faces on the current frame
    faces = face_cascade.detectMultiScale(frame)
    
    face_list=[]
    preds=[]

	# Plot rectangle around all the faces
    for (x,y,w,h) in faces:
        
        face_img = frame[y:y+h, x:x+w]
        
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        #as the model is trained in images resesized to 224x224 px , we resize the frames
        face_img = cv2.resize(face_img, (224, 224))
        
        #print(type(face_img))
        #normalized=face_img/255.0
        #reshaped=np.reshape(normalized,(1,224,224,3))
        #reshaped = np.vstack([])
        #result=model.predict(reshaped)
        #converting to float32 and making it 3D array
        face_img = img_to_array(face_img)
        face_img=np.expand_dims(face_img,axis=0)
        #face_img=face_img/255 
        #face_img = face_img.reshape(1,224,224,3)
        #face_img=np.vstack([face_img])
        face_img=preprocess_input(face_img)
        #print(face_img)
        #storing the continuous output
        #face_img=np.array(face_img)
        #face_img=generate.flow(face_img,batch_size=32)
        face_list.append(face_img)
    
        if len(face_list)>0:       
            
            preds = model.predict(face_list)
            #print(preds)
            for pred in preds:
                mask,nomask=pred
        
                if mask > nomask:
                    idx=1
                else:
                    idx=0
        
            percentage=max(mask,nomask)*100
    #creating the text format
    label = "{} : {:.2f}%".format(results[idx],percentage)
        
    #putting the text below the rectangle
    cv2.putText(frame, label, (x, y+h+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[idx], 2)            
        
    cv2.rectangle(frame, (x, y), (x + w, y + h),color[idx], 2)
        


	# Display the frame
    cv2.imshow("Video Frame", frame)
	

	# Find the key pressed
    key_pressed = cv2.waitKey(1) & 0xFF

	# If keypressed is q then quit the screen
    if key_pressed == ord('q'):
        break
    

# release the camera resource and destroy the window opened.
cap.release()
cv2.destroyAllWindows()