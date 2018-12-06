import * as tf from "@tensorflow/tfjs";
import { mobileNetClasses} from './utils/mobileNet-classes.js'


const mobileNetPath = `https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json`;

export const loadTruncatedMobileNet = async () => {
  try {
    const mobileNet = await tf.loadModel(mobileNetPath);
    // mobileNet.summary();
    const layer = mobileNet.getLayer(`conv_pw_13_relu`); //final activation layer that is not softmax
    const truncatedModel = tf.model({ inputs: mobileNet.inputs, outputs: layer.output });
    return truncatedModel;
  } catch (err) {
    console.log(err);
  }
};

const imageToTensor = (imageSrc) => {
  return new Promise((resolve) => {
    const img = new Image();
    img.src = imageSrc;
    img.onload = () => {
      const tensorImage = tf.fromPixels(img);
      return resolve(tensorImage);
    };
  })
};

// imageToTensor(testImage).then(result => console.log(result))

const cropImageTensor = (img) => {
  return tf.tidy(() => {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - size / 2;
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - size / 2;
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
  })
};

const batchImageTensor = (img) => {
  return tf.tidy(() => {
    const batched = img.expandDims(0);
    return batched.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
  })
}

/* Formats images into correctly sized tensors for the MobileNet model */
export const formatImage = async (img) => {
  const newTensor = await imageToTensor(img)
  return tf.tidy(() => { 
    const cropped = cropImageTensor(newTensor)
    newTensor.dispose();
    const resized = tf.image.resizeBilinear(cropped, [224, 224]);
    const batched = batchImageTensor(resized)
    // console.log('this is the final format of your image tensor: ', batched)
    return batched;
  })
};

/* get Xs for one image at a time --> see getXs() for batching */
export const predictFromTruncated = async (img) => {
  const truncatedModel = await loadTruncatedMobileNet()
  const imageToPredict = await formatImage(img)
  const prediction = await truncatedModel.predict(imageToPredict)
  // prediction.print(); //these are the xs
  return prediction
}

/* Predicts image content using basic MobileNet model & categories */
const predictFromMobileNet = async (img) => {
  const mobileNet = await tf.loadModel(mobileNetPath);
  const imageToPredict = await formatImage(img)
  const prediction = await mobileNet.predict(imageToPredict)
  const label = prediction.as1D().argMax().dataSync()[0]
  console.log('Predicted label key is:', label)
  console.log('Predicted class name is:', mobileNetClasses[label])
}







