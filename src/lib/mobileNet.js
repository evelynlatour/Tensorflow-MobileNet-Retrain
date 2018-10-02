import * as tf from "@tensorflow/tfjs";
import { mobileNetClasses} from './mobileNet-classes.js'

const mobileNetPath = `https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json`;
const numClasses = 5;

const loadMobileNet = async () => {
  try {
    const mobileNet = await tf.loadModel(mobileNetPath);
    // mobileNet.summary();
    const layer = mobileNet.getLayer(`conv_pw_13_relu`); //final activation layer that is not softmax
    // console.log(layer);
    const truncatedModel = tf.model({ inputs: mobileNet.inputs, outputs: layer.output });
    // console.log(truncatedModel);
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
      console.log(img.height);
      console.log(img.width);
      const tensorImage = tf.fromPixels(img);
      return resolve(tensorImage);
    };
  })
};

// imageToTensor(testImage).then(result => console.log(result))

const cropImageTensor = (img) => {
  const size = Math.min(img.shape[0], img.shape[1]);
  const centerHeight = img.shape[0] / 2;
  const beginHeight = centerHeight - size / 2;
  const centerWidth = img.shape[1] / 2;
  const beginWidth = centerWidth - size / 2;
  return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
};

const resizeImageTensor = (img) => {
  return tf.image.resizeBilinear(img, [224, 224]);
}

const batchImageTensor = (img) => {
  const batched = img.expandDims(0);
  return batched.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
}

/* Send images in to be formatted correctly for MobileNet models */
const formatImage = async (img) => {
  const newTensor = await imageToTensor(img)
  return tf.tidy(() => { 
    const cropped = cropImageTensor(newTensor)
    newTensor.dispose();
    const resized = resizeImageTensor(cropped)
    const batched = batchImageTensor(resized)
    console.log(batched)
    return batched;
  })
};

const predict = async (img) => {
  const mobileNet = await tf.loadModel(mobileNetPath);
  const imageToPredict = await formatImage(img)
  const prediction = await mobileNet.predict(imageToPredict)
  const label = prediction.as1D().argMax().dataSync()[0]
  console.log('Predicted label key is:', label)
  console.log('Predicted class name is:', mobileNetClasses[label])
}

export { formatImage, predict };
