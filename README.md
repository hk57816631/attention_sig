<h1>attention_sig</h1>

<p>Dataset should be saved in image/</p>
<ul>
  <li>the original image should be saved in image/BUS/</li>
  <li>the ground truth should be saved in image/GT/</li>
</ul>
<p>attention/generator.py is used to do pre-processing. There are two main function</p>
<ul>
  <li>the first one is used to map original image into -1 to 1</li>
  <li>the second one is used to make arguments of the image.</li>
</ul>
<p>model/unet.py is my network structure and the fuzzy part is in model/gausslayer.py. It has two type of fuzzy membership functions:</p>
<ul>
  <li>Gaussian membership function</li>
  <li>Sigmoid membership function</li>
</ul>
<p>There are also two other network structure in model/:</p>
<ul>
  <li>FCN-8s</li>
  <li>pspnet</li>
</ul>
<p>gensamples.py is used to make arguments of training samples before training.</p>
<p>test.py is used to test one testing sample.</p>
<p>test_input_patch.py is used to a set of testing samples. The names of the testing set should be given in test_file = './val.txt'. The visible segmentation result and the possibility map are given in “. png” and “. mat” files respectively.</p>
<p>train.txt is the name list of training samples.</p>
<p>wavelet.py is used to compute the wavelet transform of original images.</p>
