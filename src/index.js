import { ConsoleDisplay } from './lib/display'
import { formatImage, predict } from './lib/mobileNet'
import testImage from './images/test.jpg'
import testImage2 from './images/test2.jpg'
import colorful from './images/colorful.jpg'

// new ConsoleDisplay().display()

// formatImage(testImage)

predict(colorful)