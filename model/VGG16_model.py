from keras import applications
model = applications.VGG16(include_top=False, weights=None)  # weight='imagenet'
model.summary()