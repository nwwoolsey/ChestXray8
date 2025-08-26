remotes::install_github("rstudio/reticulate") #All of these need to only be installed once
remotes::install_github("rstudio/tensorflow")
remotes::install_github("rstudio/keras")
reticulate::install_python()
keras::install_keras()

# restart the R session, cmd+shift+0 or ctrl+shift+F10
library(keras)
library(randomcoloR)
library(tidyverse)
library(data.table)
library(caret)
library(pROC)
library(gridExtra)
library(imager)
library(tensorflow)
library(keras)
library(imager)
library(data.table)
library(dplyr)
library(tidyr)
library(stringr)
reticulate::virtualenv_install("pandas", envname = "r-tensorflow")
reticulate::py_install("pillow",env="r-reticulate")
py_require_legacy_keras()
py_require("tensorflow")
setwd('~')#Make sure this file is placed in the main folder CXR8
dat<-read.csv("Data_Entry_2017_v2020.csv")
set.seed(1)

entries <- fread( "Data_Entry_2017_v2020.csv")             # Load the data CSV File
labels_raw <- entries %>%
  select(`Patient ID`,`Image Index`, `Finding Labels`)     # adjust if column names differ
labels <- labels_raw %>%                                   # Tell dplyr to look at labels_raw
  mutate(Finding = str_split(`Finding Labels`, "\\|")) %>% # Separate multiple diagnoses
  unnest(Finding) %>%                                      # New row for each condition
  filter(Finding != "") %>%                                # Drop empty strings
  mutate(Finding = str_trim(Finding)) %>%                  # Removes white space
  pivot_wider(names_from = Finding,                        # using names from Finding column
              values_from = Finding,
              values_fill = list(Finding = 0),
              values_fn = length) %>%                      # Create a column for each condition, and 1 if present, 0 if not.
  as_tibble()

# Create stratified patient splits so individual patients don't show up in train and test data, as they are not independent. (Avoid leakage)
patients <- unique(labels$`Patient ID`)

train_pat <- createDataPartition(patients, p = 0.7, list = FALSE) #Training set is 70% of patients
val_pat   <- createDataPartition(setdiff(patients, patients[train_pat]), p = 0.15, list = FALSE) #15% is for validation during fit

train_patients <- patients[train_pat]
val_patients   <- patients[val_pat]
test_patients  <- setdiff(patients, c(train_patients, val_patients)) #Last 15% is saved as test data

# Subdivide data frame into the Training, Test, and Validation set.
train_idx <- labels %>% filter(`Patient ID` %in% train_patients) 
val_idx   <- labels %>% filter(`Patient ID` %in% val_patients)
test_idx  <- labels %>% filter(`Patient ID` %in% test_patients)

target_size <- c(224, 224) #Compress size for ease of use, could use actual image size 1024 x 1024 if you have industrial processing



train_gen <- image_data_generator(#Initialize image generation properly scaled
  rescale = 1/255,
  rotation_range = 15,
  width_shift_range = 0.05,
  height_shift_range = 0.05,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

val_gen <- image_data_generator(rescale = 1/255)

setwd("images")#Note images must be unzipped and in this folder


train_flow <- flow_images_from_dataframe(#Creates the training dataset from images
  train_idx,
  x_col = "Image Index",                 # Input is images from file
  y_col = colnames(train_idx)[-(1:3)],   # Response is the disease classifications
  target_size = target_size,
  batch_size = 32,
  class_mode = "raw",                    # returns raw vectors for multi‑label
  shuffle = TRUE
)

val_flow <- flow_images_from_dataframe(  #Creates validation training set
  val_idx,
  x_col = "Image Index",                 # Input is images from file
  y_col = colnames(train_idx)[-(1:3)],   # Response is the disease classifications
  target_size = target_size,
  batch_size = 32,
  class_mode = "raw",
  shuffle = FALSE
)
base_model <- application_densenet121(#Initialize base model using imagenet preset weights
  weights = "imagenet",
  include_top = FALSE,
  input_shape = c(224,224,3)
)


model <- keras_model(inputs = base_model$input,                                 #Initialize keras model
                     outputs = layer_dense(
                       base_model$output %>% layer_global_average_pooling_2d(), #We use 2d pooling
                       units = 15,
                       activation = "sigmoid")                                  #sigmoid activation, reLu is also possible to use
)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-4), #Compile the model
  loss = "binary_crossentropy",                     #Specify that it is a binary model and use crossentropy for loss
  metrics = list(
    metric_auc(name = "auc"),                       #We care about AUC
    metric_accuracy(name = "accuracy")              #We also care about accuracy
  )
)

summary(model) #Model summary showing that 2dpooling is trainable

callbacks_list <- list( #Callbacks in case of interruption (Save last best weights), or if the model stops improving (Terminate Early).
  callback_reduce_lr_on_plateau(patience = 3, factor = 0.5, verbose = 1),
  callback_early_stopping(patience = 7, verbose = 1, restore_best_weights = TRUE),
  callback_model_checkpoint(filepath = "best_densenet.h5",
                            save_best_only = TRUE,
                            monitor = "val_auc",
                            mode = "max",
                            verbose = 1)
)


model %>% fit(                                   #Fitting the model using the training set
  train_flow,                                    #Training set
  steps_per_epoch = ceiling(nrow(train_idx)/32), #Step sizes
  epochs = 10,                                   #Fit 10 times
  validation_data = val_flow,                    # Use Validation set for training
  callbacks = callbacks_list                     #Use our Callbacks
)



test_flow <- flow_images_from_dataframe( #Create test data set
  test_idx,
  x_col = "Image Index",
  y_col = colnames(train_idx)[-(1:3)],   # all label columns
  target_size = target_size,
  batch_size = 32,
  class_mode = "raw",
  shuffle = FALSE
)

pred_test <- model %>% predict( #Use our model to predict the test data for validation.
  test_flow,
  steps = ceiling(nrow(test_idx)/32)
)


colors<-distinctColorPalette(15)
aucs<-c()
for(i in 1:15) { 
  roc_obj <- roc(test_idx[[i+3]], pred_test[,i], quiet = TRUE)
  if(i == 1){
    plot(roc_obj,col=colors[i])
  }else{
    plot(roc_obj,add=TRUE,col=colors[i])
  }
  aucs<-cbind(aucs,auc(roc_obj))
}

names(aucs) <- colnames(test_idx[-(1:3)])
legend("bottomright",legend=names(aucs),col=colors,lty=1,lwd=3)
print(aucs)
macro_auc <- mean(aucs, na.rm = TRUE)


########################################################################

library(imager)
library(magick)
library(png)

prep_image <- function(path) {                           #Loading an image into a tensor for processing
  img  <- image_read(path) %>% image_scale("224x224")    #read image, rescale from 1024x1024 to 224x224
    raw     <- image_data(img, channels = "rgb") 
  int_px  <- as.integer(raw)                             #Convert to numbers from colors
  norm_px <- int_px / 255                                #Resize to [0,1]
  batch_px <- array_reshape(norm_px, c(1, dim(norm_px))) #Reshape into tensor, add empty dimension
  tf$convert_to_tensor(batch_px, dtype = tf$float32)     #Put into tensorflow form
}

pred_class<-11                                                      #Choose which class we want to study.
names(aucs)[pred_class]                                             #Prints which class was chosen
best_img<-which(pred_test[,pred_class]==max(pred_test[,pred_class]))#Selects image from test set with highest indicator for this class
x<-prep_image(test_idx$`Image Index`[best_img])                     #Prepare image


grad_cam <- function(model,img_array,last_conv_layer_name,target_class) {#Create heatmap for image
  grad_model <- keras_model(                                             #Load model using last convolution layer for gradient
    inputs  = model$input,
    outputs = list(
      model$get_layer(last_conv_layer_name)$output,
      model$output
    )
  )
  with(tf$GradientTape() %as% tape, {
    conv_and_preds <- grad_model(img_array)                       # Put test image into grad model
    last_conv_layer_output <- conv_and_preds[[1]]                 # Extract last_conv_layer
    preds                <- conv_and_preds[[2]]                   # Take predicted values
    class_channel <- preds[,target_class]                         # Look at predicted values for class we care about
    grads <- tape$gradient(class_channel, last_conv_layer_output) # Numerically calculate gradient over this class
    pooled_grads <- tf$reduce_mean(grads, axis = c(0L, 1L, 2L))   # Average over tensor
    last_conv_layer_output <- last_conv_layer_output[1,,,]        # Take the dimension for the last layer we care about
    heatmap <- -tf$matmul(last_conv_layer_output,tf$expand_dims(pooled_grads, -1L)) #Weight gradient over class to make heatmap
    heatmap <- tf$squeeze(heatmap)       
    heatmap <- tf$nn$relu(heatmap)                                 # keep only positive
    max_val <- tf$math$reduce_max(heatmap)                         # normalize heatmap to [0,1]
    heatmap <- tf$cond(
      tf$equal(max_val, 0L),
      function() tf$zeros_like(heatmap),
      function() heatmap / max_val
    )
  })
  # Return as a plain R matrix
  as.array(heatmap$numpy())
}

last_conv_layer_name<-"conv5_block16_concat" #This can be found manually through summary(model)

cam <- grad_cam( #Create initial heatmap
  model,
  img_array = x,
  target_class = pred_class,
  last_conv_layer_name = last_conv_layer_name
)

write_heatmap <- function(heatmap, filename, width = 224, height = 224, #Write the heatmap as a png
                          bg = "white", col = terrain.colors(12)) {
  png(filename, width = width, height = height, bg = bg) 
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev)) #Rotate if needed
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}


heatmap_highres <- tf$image$resize( #Heatmap is only 7x7, so this blends to upscale a bit
  images = tf$expand_dims(tf$constant(cam), 0L),
  size   = tf$constant(c(224L,224L),dtype=tf$int32),
  method = "lanczos3"
)


orig<-image_read(test_idx$`Image Index`[best_img])%>%image_resize("!224x224") #Read base x-ray image
orig_info<-image_info(orig)
geometry<-sprintf("%dx%d!", orig_info$width, orig_info$height)  #Saves geometry in case heatmap isnt oriented properly
pal <- col2rgb(viridis(120), alpha = TRUE)                      #Creates viridis color palette
alpha <- floor(seq(0, 255, length = ncol(pal)))                 #Layers the alpha (lower heatmap value = less opaque)

pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
write_heatmap(heatmap_highres[1,,], "heatmap_overlay.png", 
              width = 224, height = 224, bg = NA, col = pal_col)  #Save the upscaled heatmap as png

image_composite(orig,image_read("heatmap_overlay.png"),operator = "dissolve", compose_args = "55%") %>% #Lay over composite, can change % to make heatmap more opaque
  plot() 
