from keras import applications
model=applications.Xception(include_top=False,weights=None)
model.summary()