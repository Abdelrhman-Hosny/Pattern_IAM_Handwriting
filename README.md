# <u>Writer Recognition on Offline Handwriting using the IAM Dataset</u>

## <u>**Project Pipeline**</u>

1. ### <u>Preprocessing</u>

   Preprocessing consist on three main steps

   1. <u>**Smoothing and binarizing image**</u>

   2. **<u>Detecting the writing area</u>**

      This was done by utilizing **Hough Transform** to detect the lines that surrounded the handwriting in order to crop the photo.

   3. **<u>Separating each line</u>**

      We used **horizontal projection** on the writing area to separate each line on its own

   ****

2. ### **<u>Feature Extraction</u>**

   The features we extracted come from 3 papers

   1. **<u>Baseline features</u>**

      Paper: ***Writer Identification Using Text Line Based Features*** - *U.-V. Marti, R. Messerli and H. Bunke*

      ![img](https://lh6.googleusercontent.com/VBcI-nG0x93n64z-2Q0_We5huQ_yopDhdmw7g604PcwM5xQchmOIksPd4DrX6Txy51Kh8sznK428XUD3mcOeeItOMb4dg42JCaNYmQeskW0ZHNP-8n_lEi8-n4Mgpr1WaToHuwTu)

      This extracts how "tall" is the handwriting like how high does the writer write the d (topline) as can be seen the the figure above

   2. **<u>Width Between each letter and the next</u>**

      Paper: ***Writer Identification Using Text Line Based Features*** - *U.-V. Marti, R. Messerli and H. Bunke*

      **![img](https://lh6.googleusercontent.com/r81HY24wJvP1v6HSZ65vhenBr785H4MPSCP4_bqrwlrGeh4APijvDWHK4Zpn9fhOUPWBf7wYFn_m_7COWoKqQRGVQHFkzGXHgCGHpjTOSwrY7K_5kLH1MDvbxMcvBSAJ3ZgXdt0f)**

      This feature computes the distance between each line and the next

   3. <u>**Slant Feature**</u>

      Paper: **Writer Identification Using Edge-Based Directional Features** 

      ![img](https://lh4.googleusercontent.com/hpZbZGyC6ZlT9awhp039a454WWuZvwU-b9c1Xji2ahF53aGkHF0TEy1bAEdmnRhhyaS3xlFaZKZWRZKAMRnuQro-jd_qSK5z_wVz-O5fzhO0VSnyX-Wv3YGcEUyKj983vlfWhnxX)

      This feature detects the inclination of the hand writing.

   ****

3. <u>**Data Visualization**</u>

   We explored the data a bit to check whether the functions where separable and whether the data was highly complex. The data wasn't that complex and a clear decision boundary could be made which lead us to try simpler models first.

   ​									Opens example number 124 from the train set and shows features 3 and 4.

   ![img](https://lh6.googleusercontent.com/5u8oejiyx-ct8Isy5alNksoW6Y9XUaoMhHTQmKT7G-vT3cuYAwOVUZisdxCosGVY5p9YFeNwRY8TpLBXbqwAKvdBDNDWcDDTkgxki5sB8SqT5tb8QnSlLfWsHACDN9w3LFWU8sQI)

   ​									Opens example number 124 from the train set and draws features 1 , 2 and 0

   ![img](https://lh6.googleusercontent.com/5o8hsVBf8ietfBg62c4voMCs64LT2gfo90IklYo8o74kPVxZ5fXNdrRiELhyIwGrqTZX22fJw063vaM_HpIxIiQysE5qJX1U-L9tilFqMQBCdvjxuUan4kYXqbje0hG5UmLH2UAP)

   ****

4. ### **<u>Model Selection</u>**

   As discussed, we tried simpler models like linear regression and k-nearest neighbors and it got us very high results (99% acc) which was too good to be true which led us to change the model into a Random Forest Classifier.

   The Random Forest Classifier achieved a more believable accuracy of 85% and kept that accuracy when tested on handwriting outside of the dataset.

   ****

5. ### <u>Future Work</u>

   After making the enhancements , we checked out the examples that the model got wrong in the validation set and noticed that the features we added were the same as the two writers ; they had nearly the same writing size , the same white spaces between words and the same “slantness” of text . So what we could do to improve the model more is to add some other features based on what we notice was different from the wrong examples.
   ![img](https://lh4.googleusercontent.com/5thUljhE5hZ0Ak1MSAkqzqtZ7qbdDX0zs5EznYLu165kWVRWGb-LAqzHhexnfpCn-_KVEfvmq5b1abpU1jMqTtsjvIn78eyG3-q0wLUriVm6Zt661bGHMvsZDcoHh675uEfYOjLN)
   	This is a screenshot from one of the misclassified examples , from reading the papers when searching for initial features is that adding a feature that measures the size of enclosed loops could differentiate between these two images.
   If we consider how people write the letter e , not all people write the upper part of e with the same size or the same roundness and if they have little roundness in the letter e , they’ll probably not have it in the other letters that have loops which is why I think this feature would be a good addition to the model.

****

