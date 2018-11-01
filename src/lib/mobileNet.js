import * as tf from "@tensorflow/tfjs";
import { mobileNetClasses} from './mobileNet-classes.js'
import { trainingData, dataLabels } from '../images'


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
      // console.log(img.height, img.width);
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

/* Formats images into correctly sized tensors for the MobileNet model */
const formatImage = async (img) => {
  const newTensor = await imageToTensor(img)
  return tf.tidy(() => { 
    const cropped = cropImageTensor(newTensor)
    newTensor.dispose();
    const resized = resizeImageTensor(cropped)
    const batched = batchImageTensor(resized)
    // console.log('this is the final format of your image tensor: ', batched)
    return batched;
  })
};

/* Predicts image content using basic MobileNet model & categories */
const predictFromMobileNet = async (img) => {
  const mobileNet = await tf.loadModel(mobileNetPath);
  const imageToPredict = await formatImage(img)
  const prediction = await mobileNet.predict(imageToPredict)
  const label = prediction.as1D().argMax().dataSync()[0]
  console.log('Predicted label key is:', label)
  console.log('Predicted class name is:', mobileNetClasses[label])
}

/* get Xs for one image at a time --> see getXs() for batching */
export const predictFromTruncated = async (img) => {
  const truncatedModel = await loadTruncatedMobileNet()
  const imageToPredict = await formatImage(img)
  const prediction = await truncatedModel.predict(imageToPredict)
  // prediction.print(); //these are the xs
  return prediction
}

////////////////////////////// Load your own images...




/* Function to get the xs --> all the activations from the truncated MobileNet 
model after passing through our training data  */
export const getXs = async (images) => {
  try {
    let xs;
    const truncatedModel = await loadTruncatedMobileNet()
    console.log('%c Converting images to tensors...', 'color: #b159ff; font-weight: bold')
    for (let i = 0; i < images.length; i++) {
      const image = images[i];
      const imageToPredict = await formatImage(image)
      const processedImage = await truncatedModel.predict(imageToPredict)

      if (!xs) { // handle first run through
        xs = tf.keep(processedImage)
      } else { //handle all others
        const prevXs = xs;
        xs = tf.keep(prevXs.concat(processedImage, 0));
        prevXs.dispose();
      }
    }
    // xs.print();
    console.log('%c These are your xs: ', 'color: #ffb85b; font-weight: bold' , xs) // shape of Xs: [10,7,7,256] where 10 is the batch size (from testing 5 red & 5 blue)
    return xs;
  } catch (err) {
    console.log(err)
  }
}



/* Get the ys --> the labels for all of the collected data as a "one hot" representation */
// labelIndex is the index at which there is a 1
// oneHot(3, 5) => [0,0,0,1,0]
export const oneHot = (labelIndex, numClasses) => { 
  return tf.tidy(() => tf.oneHot(tf.tensor1d([labelIndex]).toInt(), numClasses));
};

// Use only if you do not have a predefined label object (e.g. for an array like red/blue testing)
export const labelObjectMaker = (labels) => {
  let labelObj = {};
  labels.forEach(label => {
    if (labelObj[label] === undefined) {
      labelObj[label] = Object.keys(labelObj).length
    }
  })
  console.log('this is the label object key: ', labelObj)
  return labelObj
}

export const getYs = (labels, labelKey) => {
  // const classes = labelObjectMaker(labels); // see note above on when to use this func
  const classes = labelKey
  const classLength = Object.keys(classes).length;
  let ys;

  labels.forEach(label => {
    const labelIndex = classes[label];
    const y = oneHot(labelIndex, classLength);
    if (!ys) {
      ys = tf.keep(y)
    } else {
      const prevYs = ys;
      ys = tf.keep(prevYs.concat(y, 0))
      prevYs.dispose();
      y.dispose();
    }
  })
  // ys.print();
  console.log('%c These are your ys: ', 'color: #ffb85b; font-weight: bold' , ys)
  return ys;
}


/* In order to train your model, you'll need to feed it these xs and ys */

export const listModelsInLocalStorage = async () => {
  console.log('%c Models available in the browser...', 'color: #63DFFF')
  console.log(await tf.io.listModels())
}

export const loadCustomModel = async (modelName) => {
  // List models stored in browser
  console.log('%c Models available in the browser...', 'color: #63DFFF')
  console.log(await tf.io.listModels())
  console.log(`%c Loading ${modelName}`, 'color: #49FFE0')
  const customModel = await tf.loadModel(`indexeddb://${modelName}`)
  return customModel
} 
