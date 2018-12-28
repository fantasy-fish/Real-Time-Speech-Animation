# Real-Time-Speech-Animation
## Quick Start
Install all the requirements by<br/>
```python
pip install -r requirements.txt
```
Then run window.py by<br/>
```python
python window.py
```
After the window pops up, you need to first sample the background noise and then press the start button.<br/>
## Demo
Below is a demo video of the current version of system when it's connected with the microphone. There's a huge delay for the animation mainly due to the denoising pipeline. Currently, I store every piece of audio segment to file and apply the denoising method to each of them on the fly. It can be improved by trying some real-time denoising method.<br/>
[![IMAGE ALT TEXT](http://img.youtube.com/vi/7A9vIIs5Y7k/0.jpg)](https://www.youtube.com/watch?v=7A9vIIs5Y7k)<br/>
For comparison, below shows a previous version of system when not connected with the microphone so that the denoising process is done in advance instead of on the fly.<br/>
[![IMAGE ALT TEXT](http://img.youtube.com/vi/fbL0ZdfddyI/0.jpg)](https://www.youtube.com/watch?v=fbL0ZdfddyI)
