# ERA-SESSION13 YoloV3 with Pytorch Lightning & Gradio

HF Link:

### Achieved:
1. Training Loss: 3.680
2. Validation Loss: 4.940
3. Class accuracy: 81.601883%
4. No obj accuracy: 97.991463%
5. Obj accuracy: 75.976616%
6. MAP: 0.4366795

### Tasks:
1. :heavy_check_mark: Move the code to PytorchLightning
2. :heavy_check_mark: Train the model to reach such that all of these are true:
    - Class accuracy is more than 75%
    - No Obj accuracy of more than 95%
    - Object Accuracy of more than 70% (assuming you had to reduce the kernel numbers, else 80/98/78)
    - Ideally trained till 40 epochs
3. :heavy_check_mark: Add these training features:
    - Add multi-resolution training - the code shared trains only on one resolution 416
    - Add Implement Mosaic Augmentation only 75% of the times
    - Train on float16
    - GradCam must be implemented.
4. :heavy_check_mark: Things that are allowed due to HW constraints:
    - Change of batch size
    - Change of resolution
    - Change of OCP parameters
5. Once done:
    - Move the app to HuggingFace Spaces
    - Allow custom upload of images
    - Share some samples from the existing dataset
    - Show the GradCAM output for the image that the user uploads as well as for the samples.
6. Mention things like:
    - classes that your model support
    - link to the actual model
7. Assignment:
    - Share HuggingFace App Link
    - Share LightningCode Link on Github
    - Share notebook link (with logs) on GitHub
  
### Results
![image](https://github.com/RaviNaik/ERA-SESSION13/blob/main/yolo_results.png)

### Model Summary
```python
    | Name                       | Type              | Params
-------------------------------------------------------------------
0   | loss_fn                    | YoloLoss          | 0     
1   | loss_fn.mse                | MSELoss           | 0     
2   | loss_fn.bce                | BCEWithLogitsLoss | 0     
3   | loss_fn.entropy            | CrossEntropyLoss  | 0     
4   | loss_fn.sigmoid            | Sigmoid           | 0     
5   | layers                     | ModuleList        | 61.6 M
6   | layers.0                   | CNNBlock          | 928   
7   | layers.0.conv              | Conv2d            | 864   
8   | layers.0.bn                | BatchNorm2d       | 64    
9   | layers.0.leaky             | LeakyReLU         | 0     
10  | layers.1                   | CNNBlock          | 18.6 K
11  | layers.1.conv              | Conv2d            | 18.4 K
12  | layers.1.bn                | BatchNorm2d       | 128   
13  | layers.1.leaky             | LeakyReLU         | 0     
14  | layers.2                   | ResidualBlock     | 20.7 K
15  | layers.2.layers            | ModuleList        | 20.7 K
16  | layers.2.layers.0          | Sequential        | 20.7 K
17  | layers.2.layers.0.0        | CNNBlock          | 2.1 K 
18  | layers.2.layers.0.0.conv   | Conv2d            | 2.0 K 
19  | layers.2.layers.0.0.bn     | BatchNorm2d       | 64    
20  | layers.2.layers.0.0.leaky  | LeakyReLU         | 0     
21  | layers.2.layers.0.1        | CNNBlock          | 18.6 K
22  | layers.2.layers.0.1.conv   | Conv2d            | 18.4 K
23  | layers.2.layers.0.1.bn     | BatchNorm2d       | 128   
24  | layers.2.layers.0.1.leaky  | LeakyReLU         | 0     
25  | layers.3                   | CNNBlock          | 74.0 K
26  | layers.3.conv              | Conv2d            | 73.7 K
27  | layers.3.bn                | BatchNorm2d       | 256   
28  | layers.3.leaky             | LeakyReLU         | 0     
29  | layers.4                   | ResidualBlock     | 164 K 
30  | layers.4.layers            | ModuleList        | 164 K 
31  | layers.4.layers.0          | Sequential        | 82.3 K
32  | layers.4.layers.0.0        | CNNBlock          | 8.3 K 
33  | layers.4.layers.0.0.conv   | Conv2d            | 8.2 K 
34  | layers.4.layers.0.0.bn     | BatchNorm2d       | 128   
35  | layers.4.layers.0.0.leaky  | LeakyReLU         | 0     
36  | layers.4.layers.0.1        | CNNBlock          | 74.0 K
37  | layers.4.layers.0.1.conv   | Conv2d            | 73.7 K
38  | layers.4.layers.0.1.bn     | BatchNorm2d       | 256   
39  | layers.4.layers.0.1.leaky  | LeakyReLU         | 0     
40  | layers.4.layers.1          | Sequential        | 82.3 K
41  | layers.4.layers.1.0        | CNNBlock          | 8.3 K 
42  | layers.4.layers.1.0.conv   | Conv2d            | 8.2 K 
43  | layers.4.layers.1.0.bn     | BatchNorm2d       | 128   
44  | layers.4.layers.1.0.leaky  | LeakyReLU         | 0     
45  | layers.4.layers.1.1        | CNNBlock          | 74.0 K
46  | layers.4.layers.1.1.conv   | Conv2d            | 73.7 K
47  | layers.4.layers.1.1.bn     | BatchNorm2d       | 256   
48  | layers.4.layers.1.1.leaky  | LeakyReLU         | 0     
49  | layers.5                   | CNNBlock          | 295 K 
50  | layers.5.conv              | Conv2d            | 294 K 
51  | layers.5.bn                | BatchNorm2d       | 512   
52  | layers.5.leaky             | LeakyReLU         | 0     
53  | layers.6                   | ResidualBlock     | 2.6 M 
54  | layers.6.layers            | ModuleList        | 2.6 M 
55  | layers.6.layers.0          | Sequential        | 328 K 
56  | layers.6.layers.0.0        | CNNBlock          | 33.0 K
57  | layers.6.layers.0.0.conv   | Conv2d            | 32.8 K
58  | layers.6.layers.0.0.bn     | BatchNorm2d       | 256   
59  | layers.6.layers.0.0.leaky  | LeakyReLU         | 0     
60  | layers.6.layers.0.1        | CNNBlock          | 295 K 
61  | layers.6.layers.0.1.conv   | Conv2d            | 294 K 
62  | layers.6.layers.0.1.bn     | BatchNorm2d       | 512   
63  | layers.6.layers.0.1.leaky  | LeakyReLU         | 0     
64  | layers.6.layers.1          | Sequential        | 328 K 
65  | layers.6.layers.1.0        | CNNBlock          | 33.0 K
66  | layers.6.layers.1.0.conv   | Conv2d            | 32.8 K
67  | layers.6.layers.1.0.bn     | BatchNorm2d       | 256   
68  | layers.6.layers.1.0.leaky  | LeakyReLU         | 0     
69  | layers.6.layers.1.1        | CNNBlock          | 295 K 
70  | layers.6.layers.1.1.conv   | Conv2d            | 294 K 
71  | layers.6.layers.1.1.bn     | BatchNorm2d       | 512   
72  | layers.6.layers.1.1.leaky  | LeakyReLU         | 0     
73  | layers.6.layers.2          | Sequential        | 328 K 
74  | layers.6.layers.2.0        | CNNBlock          | 33.0 K
75  | layers.6.layers.2.0.conv   | Conv2d            | 32.8 K
76  | layers.6.layers.2.0.bn     | BatchNorm2d       | 256   
77  | layers.6.layers.2.0.leaky  | LeakyReLU         | 0     
78  | layers.6.layers.2.1        | CNNBlock          | 295 K 
79  | layers.6.layers.2.1.conv   | Conv2d            | 294 K 
80  | layers.6.layers.2.1.bn     | BatchNorm2d       | 512   
81  | layers.6.layers.2.1.leaky  | LeakyReLU         | 0     
82  | layers.6.layers.3          | Sequential        | 328 K 
83  | layers.6.layers.3.0        | CNNBlock          | 33.0 K
84  | layers.6.layers.3.0.conv   | Conv2d            | 32.8 K
85  | layers.6.layers.3.0.bn     | BatchNorm2d       | 256   
86  | layers.6.layers.3.0.leaky  | LeakyReLU         | 0     
87  | layers.6.layers.3.1        | CNNBlock          | 295 K 
88  | layers.6.layers.3.1.conv   | Conv2d            | 294 K 
89  | layers.6.layers.3.1.bn     | BatchNorm2d       | 512   
90  | layers.6.layers.3.1.leaky  | LeakyReLU         | 0     
91  | layers.6.layers.4          | Sequential        | 328 K 
92  | layers.6.layers.4.0        | CNNBlock          | 33.0 K
93  | layers.6.layers.4.0.conv   | Conv2d            | 32.8 K
94  | layers.6.layers.4.0.bn     | BatchNorm2d       | 256   
95  | layers.6.layers.4.0.leaky  | LeakyReLU         | 0     
96  | layers.6.layers.4.1        | CNNBlock          | 295 K 
97  | layers.6.layers.4.1.conv   | Conv2d            | 294 K 
98  | layers.6.layers.4.1.bn     | BatchNorm2d       | 512   
99  | layers.6.layers.4.1.leaky  | LeakyReLU         | 0     
100 | layers.6.layers.5          | Sequential        | 328 K 
101 | layers.6.layers.5.0        | CNNBlock          | 33.0 K
102 | layers.6.layers.5.0.conv   | Conv2d            | 32.8 K
103 | layers.6.layers.5.0.bn     | BatchNorm2d       | 256   
104 | layers.6.layers.5.0.leaky  | LeakyReLU         | 0     
105 | layers.6.layers.5.1        | CNNBlock          | 295 K 
106 | layers.6.layers.5.1.conv   | Conv2d            | 294 K 
107 | layers.6.layers.5.1.bn     | BatchNorm2d       | 512   
108 | layers.6.layers.5.1.leaky  | LeakyReLU         | 0     
109 | layers.6.layers.6          | Sequential        | 328 K 
110 | layers.6.layers.6.0        | CNNBlock          | 33.0 K
111 | layers.6.layers.6.0.conv   | Conv2d            | 32.8 K
112 | layers.6.layers.6.0.bn     | BatchNorm2d       | 256   
113 | layers.6.layers.6.0.leaky  | LeakyReLU         | 0     
114 | layers.6.layers.6.1        | CNNBlock          | 295 K 
115 | layers.6.layers.6.1.conv   | Conv2d            | 294 K 
116 | layers.6.layers.6.1.bn     | BatchNorm2d       | 512   
117 | layers.6.layers.6.1.leaky  | LeakyReLU         | 0     
118 | layers.6.layers.7          | Sequential        | 328 K 
119 | layers.6.layers.7.0        | CNNBlock          | 33.0 K
120 | layers.6.layers.7.0.conv   | Conv2d            | 32.8 K
121 | layers.6.layers.7.0.bn     | BatchNorm2d       | 256   
122 | layers.6.layers.7.0.leaky  | LeakyReLU         | 0     
123 | layers.6.layers.7.1        | CNNBlock          | 295 K 
124 | layers.6.layers.7.1.conv   | Conv2d            | 294 K 
125 | layers.6.layers.7.1.bn     | BatchNorm2d       | 512   
126 | layers.6.layers.7.1.leaky  | LeakyReLU         | 0     
127 | layers.7                   | CNNBlock          | 1.2 M 
128 | layers.7.conv              | Conv2d            | 1.2 M 
129 | layers.7.bn                | BatchNorm2d       | 1.0 K 
130 | layers.7.leaky             | LeakyReLU         | 0     
131 | layers.8                   | ResidualBlock     | 10.5 M
132 | layers.8.layers            | ModuleList        | 10.5 M
133 | layers.8.layers.0          | Sequential        | 1.3 M 
134 | layers.8.layers.0.0        | CNNBlock          | 131 K 
135 | layers.8.layers.0.0.conv   | Conv2d            | 131 K 
136 | layers.8.layers.0.0.bn     | BatchNorm2d       | 512   
137 | layers.8.layers.0.0.leaky  | LeakyReLU         | 0     
138 | layers.8.layers.0.1        | CNNBlock          | 1.2 M 
139 | layers.8.layers.0.1.conv   | Conv2d            | 1.2 M 
140 | layers.8.layers.0.1.bn     | BatchNorm2d       | 1.0 K 
141 | layers.8.layers.0.1.leaky  | LeakyReLU         | 0     
142 | layers.8.layers.1          | Sequential        | 1.3 M 
143 | layers.8.layers.1.0        | CNNBlock          | 131 K 
144 | layers.8.layers.1.0.conv   | Conv2d            | 131 K 
145 | layers.8.layers.1.0.bn     | BatchNorm2d       | 512   
146 | layers.8.layers.1.0.leaky  | LeakyReLU         | 0     
147 | layers.8.layers.1.1        | CNNBlock          | 1.2 M 
148 | layers.8.layers.1.1.conv   | Conv2d            | 1.2 M 
149 | layers.8.layers.1.1.bn     | BatchNorm2d       | 1.0 K 
150 | layers.8.layers.1.1.leaky  | LeakyReLU         | 0     
151 | layers.8.layers.2          | Sequential        | 1.3 M 
152 | layers.8.layers.2.0        | CNNBlock          | 131 K 
153 | layers.8.layers.2.0.conv   | Conv2d            | 131 K 
154 | layers.8.layers.2.0.bn     | BatchNorm2d       | 512   
155 | layers.8.layers.2.0.leaky  | LeakyReLU         | 0     
156 | layers.8.layers.2.1        | CNNBlock          | 1.2 M 
157 | layers.8.layers.2.1.conv   | Conv2d            | 1.2 M 
158 | layers.8.layers.2.1.bn     | BatchNorm2d       | 1.0 K 
159 | layers.8.layers.2.1.leaky  | LeakyReLU         | 0     
160 | layers.8.layers.3          | Sequential        | 1.3 M 
161 | layers.8.layers.3.0        | CNNBlock          | 131 K 
162 | layers.8.layers.3.0.conv   | Conv2d            | 131 K 
163 | layers.8.layers.3.0.bn     | BatchNorm2d       | 512   
164 | layers.8.layers.3.0.leaky  | LeakyReLU         | 0     
165 | layers.8.layers.3.1        | CNNBlock          | 1.2 M 
166 | layers.8.layers.3.1.conv   | Conv2d            | 1.2 M 
167 | layers.8.layers.3.1.bn     | BatchNorm2d       | 1.0 K 
168 | layers.8.layers.3.1.leaky  | LeakyReLU         | 0     
169 | layers.8.layers.4          | Sequential        | 1.3 M 
170 | layers.8.layers.4.0        | CNNBlock          | 131 K 
171 | layers.8.layers.4.0.conv   | Conv2d            | 131 K 
172 | layers.8.layers.4.0.bn     | BatchNorm2d       | 512   
173 | layers.8.layers.4.0.leaky  | LeakyReLU         | 0     
174 | layers.8.layers.4.1        | CNNBlock          | 1.2 M 
175 | layers.8.layers.4.1.conv   | Conv2d            | 1.2 M 
176 | layers.8.layers.4.1.bn     | BatchNorm2d       | 1.0 K 
177 | layers.8.layers.4.1.leaky  | LeakyReLU         | 0     
178 | layers.8.layers.5          | Sequential        | 1.3 M 
179 | layers.8.layers.5.0        | CNNBlock          | 131 K 
180 | layers.8.layers.5.0.conv   | Conv2d            | 131 K 
181 | layers.8.layers.5.0.bn     | BatchNorm2d       | 512   
182 | layers.8.layers.5.0.leaky  | LeakyReLU         | 0     
183 | layers.8.layers.5.1        | CNNBlock          | 1.2 M 
184 | layers.8.layers.5.1.conv   | Conv2d            | 1.2 M 
185 | layers.8.layers.5.1.bn     | BatchNorm2d       | 1.0 K 
186 | layers.8.layers.5.1.leaky  | LeakyReLU         | 0     
187 | layers.8.layers.6          | Sequential        | 1.3 M 
188 | layers.8.layers.6.0        | CNNBlock          | 131 K 
189 | layers.8.layers.6.0.conv   | Conv2d            | 131 K 
190 | layers.8.layers.6.0.bn     | BatchNorm2d       | 512   
191 | layers.8.layers.6.0.leaky  | LeakyReLU         | 0     
192 | layers.8.layers.6.1        | CNNBlock          | 1.2 M 
193 | layers.8.layers.6.1.conv   | Conv2d            | 1.2 M 
194 | layers.8.layers.6.1.bn     | BatchNorm2d       | 1.0 K 
195 | layers.8.layers.6.1.leaky  | LeakyReLU         | 0     
196 | layers.8.layers.7          | Sequential        | 1.3 M 
197 | layers.8.layers.7.0        | CNNBlock          | 131 K 
198 | layers.8.layers.7.0.conv   | Conv2d            | 131 K 
199 | layers.8.layers.7.0.bn     | BatchNorm2d       | 512   
200 | layers.8.layers.7.0.leaky  | LeakyReLU         | 0     
201 | layers.8.layers.7.1        | CNNBlock          | 1.2 M 
202 | layers.8.layers.7.1.conv   | Conv2d            | 1.2 M 
203 | layers.8.layers.7.1.bn     | BatchNorm2d       | 1.0 K 
204 | layers.8.layers.7.1.leaky  | LeakyReLU         | 0     
205 | layers.9                   | CNNBlock          | 4.7 M 
206 | layers.9.conv              | Conv2d            | 4.7 M 
207 | layers.9.bn                | BatchNorm2d       | 2.0 K 
208 | layers.9.leaky             | LeakyReLU         | 0     
209 | layers.10                  | ResidualBlock     | 21.0 M
210 | layers.10.layers           | ModuleList        | 21.0 M
211 | layers.10.layers.0         | Sequential        | 5.2 M 
212 | layers.10.layers.0.0       | CNNBlock          | 525 K 
213 | layers.10.layers.0.0.conv  | Conv2d            | 524 K 
214 | layers.10.layers.0.0.bn    | BatchNorm2d       | 1.0 K 
215 | layers.10.layers.0.0.leaky | LeakyReLU         | 0     
216 | layers.10.layers.0.1       | CNNBlock          | 4.7 M 
217 | layers.10.layers.0.1.conv  | Conv2d            | 4.7 M 
218 | layers.10.layers.0.1.bn    | BatchNorm2d       | 2.0 K 
219 | layers.10.layers.0.1.leaky | LeakyReLU         | 0     
220 | layers.10.layers.1         | Sequential        | 5.2 M 
221 | layers.10.layers.1.0       | CNNBlock          | 525 K 
222 | layers.10.layers.1.0.conv  | Conv2d            | 524 K 
223 | layers.10.layers.1.0.bn    | BatchNorm2d       | 1.0 K 
224 | layers.10.layers.1.0.leaky | LeakyReLU         | 0     
225 | layers.10.layers.1.1       | CNNBlock          | 4.7 M 
226 | layers.10.layers.1.1.conv  | Conv2d            | 4.7 M 
227 | layers.10.layers.1.1.bn    | BatchNorm2d       | 2.0 K 
228 | layers.10.layers.1.1.leaky | LeakyReLU         | 0     
229 | layers.10.layers.2         | Sequential        | 5.2 M 
230 | layers.10.layers.2.0       | CNNBlock          | 525 K 
231 | layers.10.layers.2.0.conv  | Conv2d            | 524 K 
232 | layers.10.layers.2.0.bn    | BatchNorm2d       | 1.0 K 
233 | layers.10.layers.2.0.leaky | LeakyReLU         | 0     
234 | layers.10.layers.2.1       | CNNBlock          | 4.7 M 
235 | layers.10.layers.2.1.conv  | Conv2d            | 4.7 M 
236 | layers.10.layers.2.1.bn    | BatchNorm2d       | 2.0 K 
237 | layers.10.layers.2.1.leaky | LeakyReLU         | 0     
238 | layers.10.layers.3         | Sequential        | 5.2 M 
239 | layers.10.layers.3.0       | CNNBlock          | 525 K 
240 | layers.10.layers.3.0.conv  | Conv2d            | 524 K 
241 | layers.10.layers.3.0.bn    | BatchNorm2d       | 1.0 K 
242 | layers.10.layers.3.0.leaky | LeakyReLU         | 0     
243 | layers.10.layers.3.1       | CNNBlock          | 4.7 M 
244 | layers.10.layers.3.1.conv  | Conv2d            | 4.7 M 
245 | layers.10.layers.3.1.bn    | BatchNorm2d       | 2.0 K 
246 | layers.10.layers.3.1.leaky | LeakyReLU         | 0     
247 | layers.11                  | CNNBlock          | 525 K 
248 | layers.11.conv             | Conv2d            | 524 K 
249 | layers.11.bn               | BatchNorm2d       | 1.0 K 
250 | layers.11.leaky            | LeakyReLU         | 0     
251 | layers.12                  | CNNBlock          | 4.7 M 
252 | layers.12.conv             | Conv2d            | 4.7 M 
253 | layers.12.bn               | BatchNorm2d       | 2.0 K 
254 | layers.12.leaky            | LeakyReLU         | 0     
255 | layers.13                  | ResidualBlock     | 5.2 M 
256 | layers.13.layers           | ModuleList        | 5.2 M 
257 | layers.13.layers.0         | Sequential        | 5.2 M 
258 | layers.13.layers.0.0       | CNNBlock          | 525 K 
259 | layers.13.layers.0.0.conv  | Conv2d            | 524 K 
260 | layers.13.layers.0.0.bn    | BatchNorm2d       | 1.0 K 
261 | layers.13.layers.0.0.leaky | LeakyReLU         | 0     
262 | layers.13.layers.0.1       | CNNBlock          | 4.7 M 
263 | layers.13.layers.0.1.conv  | Conv2d            | 4.7 M 
264 | layers.13.layers.0.1.bn    | BatchNorm2d       | 2.0 K 
265 | layers.13.layers.0.1.leaky | LeakyReLU         | 0     
266 | layers.14                  | CNNBlock          | 525 K 
267 | layers.14.conv             | Conv2d            | 524 K 
268 | layers.14.bn               | BatchNorm2d       | 1.0 K 
269 | layers.14.leaky            | LeakyReLU         | 0     
270 | layers.15                  | ScalePrediction   | 4.8 M 
271 | layers.15.pred             | Sequential        | 4.8 M 
272 | layers.15.pred.0           | CNNBlock          | 4.7 M 
273 | layers.15.pred.0.conv      | Conv2d            | 4.7 M 
274 | layers.15.pred.0.bn        | BatchNorm2d       | 2.0 K 
275 | layers.15.pred.0.leaky     | LeakyReLU         | 0     
276 | layers.15.pred.1           | CNNBlock          | 77.0 K
277 | layers.15.pred.1.conv      | Conv2d            | 76.9 K
278 | layers.15.pred.1.bn        | BatchNorm2d       | 150   
279 | layers.15.pred.1.leaky     | LeakyReLU         | 0     
280 | layers.16                  | CNNBlock          | 131 K 
281 | layers.16.conv             | Conv2d            | 131 K 
282 | layers.16.bn               | BatchNorm2d       | 512   
283 | layers.16.leaky            | LeakyReLU         | 0     
284 | layers.17                  | Upsample          | 0     
285 | layers.18                  | CNNBlock          | 197 K 
286 | layers.18.conv             | Conv2d            | 196 K 
287 | layers.18.bn               | BatchNorm2d       | 512   
288 | layers.18.leaky            | LeakyReLU         | 0     
289 | layers.19                  | CNNBlock          | 1.2 M 
290 | layers.19.conv             | Conv2d            | 1.2 M 
291 | layers.19.bn               | BatchNorm2d       | 1.0 K 
292 | layers.19.leaky            | LeakyReLU         | 0     
293 | layers.20                  | ResidualBlock     | 1.3 M 
294 | layers.20.layers           | ModuleList        | 1.3 M 
295 | layers.20.layers.0         | Sequential        | 1.3 M 
296 | layers.20.layers.0.0       | CNNBlock          | 131 K 
297 | layers.20.layers.0.0.conv  | Conv2d            | 131 K 
298 | layers.20.layers.0.0.bn    | BatchNorm2d       | 512   
299 | layers.20.layers.0.0.leaky | LeakyReLU         | 0     
300 | layers.20.layers.0.1       | CNNBlock          | 1.2 M 
301 | layers.20.layers.0.1.conv  | Conv2d            | 1.2 M 
302 | layers.20.layers.0.1.bn    | BatchNorm2d       | 1.0 K 
303 | layers.20.layers.0.1.leaky | LeakyReLU         | 0     
304 | layers.21                  | CNNBlock          | 131 K 
305 | layers.21.conv             | Conv2d            | 131 K 
306 | layers.21.bn               | BatchNorm2d       | 512   
307 | layers.21.leaky            | LeakyReLU         | 0     
308 | layers.22                  | ScalePrediction   | 1.2 M 
309 | layers.22.pred             | Sequential        | 1.2 M 
310 | layers.22.pred.0           | CNNBlock          | 1.2 M 
311 | layers.22.pred.0.conv      | Conv2d            | 1.2 M 
312 | layers.22.pred.0.bn        | BatchNorm2d       | 1.0 K 
313 | layers.22.pred.0.leaky     | LeakyReLU         | 0     
314 | layers.22.pred.1           | CNNBlock          | 38.6 K
315 | layers.22.pred.1.conv      | Conv2d            | 38.5 K
316 | layers.22.pred.1.bn        | BatchNorm2d       | 150   
317 | layers.22.pred.1.leaky     | LeakyReLU         | 0     
318 | layers.23                  | CNNBlock          | 33.0 K
319 | layers.23.conv             | Conv2d            | 32.8 K
320 | layers.23.bn               | BatchNorm2d       | 256   
321 | layers.23.leaky            | LeakyReLU         | 0     
322 | layers.24                  | Upsample          | 0     
323 | layers.25                  | CNNBlock          | 49.4 K
324 | layers.25.conv             | Conv2d            | 49.2 K
325 | layers.25.bn               | BatchNorm2d       | 256   
326 | layers.25.leaky            | LeakyReLU         | 0     
327 | layers.26                  | CNNBlock          | 295 K 
328 | layers.26.conv             | Conv2d            | 294 K 
329 | layers.26.bn               | BatchNorm2d       | 512   
330 | layers.26.leaky            | LeakyReLU         | 0     
331 | layers.27                  | ResidualBlock     | 328 K 
332 | layers.27.layers           | ModuleList        | 328 K 
333 | layers.27.layers.0         | Sequential        | 328 K 
334 | layers.27.layers.0.0       | CNNBlock          | 33.0 K
335 | layers.27.layers.0.0.conv  | Conv2d            | 32.8 K
336 | layers.27.layers.0.0.bn    | BatchNorm2d       | 256   
337 | layers.27.layers.0.0.leaky | LeakyReLU         | 0     
338 | layers.27.layers.0.1       | CNNBlock          | 295 K 
339 | layers.27.layers.0.1.conv  | Conv2d            | 294 K 
340 | layers.27.layers.0.1.bn    | BatchNorm2d       | 512   
341 | layers.27.layers.0.1.leaky | LeakyReLU         | 0     
342 | layers.28                  | CNNBlock          | 33.0 K
343 | layers.28.conv             | Conv2d            | 32.8 K
344 | layers.28.bn               | BatchNorm2d       | 256   
345 | layers.28.leaky            | LeakyReLU         | 0     
346 | layers.29                  | ScalePrediction   | 314 K 
347 | layers.29.pred             | Sequential        | 314 K 
348 | layers.29.pred.0           | CNNBlock          | 295 K 
349 | layers.29.pred.0.conv      | Conv2d            | 294 K 
350 | layers.29.pred.0.bn        | BatchNorm2d       | 512   
351 | layers.29.pred.0.leaky     | LeakyReLU         | 0     
352 | layers.29.pred.1           | CNNBlock          | 19.4 K
353 | layers.29.pred.1.conv      | Conv2d            | 19.3 K
354 | layers.29.pred.1.bn        | BatchNorm2d       | 150   
355 | layers.29.pred.1.leaky     | LeakyReLU         | 0     
-------------------------------------------------------------------
61.6 M    Trainable params
0         Non-trainable params
61.6 M    Total params
246.506   Total estimated model params size (MB)
```

### LR Finder
![image](https://github.com/RaviNaik/ERA-SESSION13/assets/23289802/a6d64f13-a7b7-4e17-abfc-3ec86e84b710)

### Loss & Accuracy
Training & Validation Loss:
![image](https://github.com/RaviNaik/ERA-SESSION13/assets/23289802/9391157e-a889-480d-b233-b72e86745245)

Testing Accuracy:
```python
0%|          | 0/39 [00:00<?, ?it/s]
  3%|▎         | 1/39 [00:05<03:24,  5.37s/it]
  5%|▌         | 2/39 [00:11<03:32,  5.75s/it]
  8%|▊         | 3/39 [00:16<03:14,  5.41s/it]
 10%|█         | 4/39 [00:21<03:06,  5.33s/it]
 13%|█▎        | 5/39 [00:26<02:55,  5.17s/it]
 15%|█▌        | 6/39 [00:31<02:50,  5.16s/it]
 18%|█▊        | 7/39 [00:36<02:43,  5.11s/it]
 21%|██        | 8/39 [00:42<02:48,  5.43s/it]
 23%|██▎       | 9/39 [00:48<02:44,  5.47s/it]
 26%|██▌       | 10/39 [00:54<02:41,  5.58s/it]
 28%|██▊       | 11/39 [00:59<02:36,  5.59s/it]
 31%|███       | 12/39 [01:05<02:35,  5.77s/it]
 33%|███▎      | 13/39 [01:11<02:28,  5.70s/it]
 36%|███▌      | 14/39 [01:16<02:15,  5.42s/it]
 38%|███▊      | 15/39 [01:21<02:07,  5.30s/it]
 41%|████      | 16/39 [01:26<02:02,  5.34s/it]
 44%|████▎     | 17/39 [01:31<01:54,  5.23s/it]
 46%|████▌     | 18/39 [01:36<01:49,  5.22s/it]
 49%|████▊     | 19/39 [01:42<01:43,  5.20s/it]
 51%|█████▏    | 20/39 [01:46<01:33,  4.94s/it]
 54%|█████▍    | 21/39 [01:50<01:23,  4.64s/it]
 56%|█████▋    | 22/39 [01:54<01:14,  4.41s/it]
 59%|█████▉    | 23/39 [01:57<01:03,  3.96s/it]
 62%|██████▏   | 24/39 [02:00<00:54,  3.66s/it]
 64%|██████▍   | 25/39 [02:04<00:55,  3.94s/it]
 67%|██████▋   | 26/39 [02:10<00:56,  4.38s/it]
 69%|██████▉   | 27/39 [02:14<00:53,  4.47s/it]
 72%|███████▏  | 28/39 [02:20<00:52,  4.77s/it]
 74%|███████▍  | 29/39 [02:25<00:50,  5.04s/it]
 77%|███████▋  | 30/39 [02:31<00:47,  5.25s/it]
 79%|███████▉  | 31/39 [02:37<00:42,  5.36s/it]
 82%|████████▏ | 32/39 [02:42<00:38,  5.43s/it]
 85%|████████▍ | 33/39 [02:47<00:31,  5.24s/it]
 87%|████████▋ | 34/39 [02:53<00:26,  5.29s/it]
 90%|████████▉ | 35/39 [02:58<00:21,  5.32s/it]
 92%|█████████▏| 36/39 [03:03<00:15,  5.23s/it]
 95%|█████████▍| 37/39 [03:08<00:10,  5.26s/it]
 97%|█████████▋| 38/39 [03:14<00:05,  5.32s/it]
100%|██████████| 39/39 [03:17<00:00,  5.07s/it]
Class accuracy is: 81.601883%
No obj accuracy is: 97.991463%
Obj accuracy is: 75.976616%
```
### MAP Calculations
```python
0%|          | 0/39 [00:00<?, ?it/s]
  3%|▎         | 1/39 [00:40<25:35, 40.40s/it]
  5%|▌         | 2/39 [01:24<26:05, 42.31s/it]
  8%|▊         | 3/39 [02:01<24:02, 40.07s/it]
 10%|█         | 4/39 [02:40<23:04, 39.57s/it]
 13%|█▎        | 5/39 [03:36<25:45, 45.46s/it]
 15%|█▌        | 6/39 [04:20<24:45, 45.00s/it]
 18%|█▊        | 7/39 [05:03<23:37, 44.29s/it]
 21%|██        | 8/39 [05:47<22:55, 44.36s/it]
 23%|██▎       | 9/39 [06:33<22:25, 44.84s/it]
 26%|██▌       | 10/39 [07:06<19:54, 41.20s/it]
 28%|██▊       | 11/39 [07:58<20:45, 44.49s/it]
 31%|███       | 12/39 [08:36<19:10, 42.60s/it]
 33%|███▎      | 13/39 [09:20<18:33, 42.81s/it]
 36%|███▌      | 14/39 [10:01<17:43, 42.53s/it]
 38%|███▊      | 15/39 [10:42<16:49, 42.04s/it]
 41%|████      | 16/39 [11:25<16:10, 42.18s/it]
 44%|████▎     | 17/39 [12:12<16:02, 43.73s/it]
 46%|████▌     | 18/39 [12:56<15:20, 43.83s/it]
 49%|████▊     | 19/39 [13:36<14:12, 42.64s/it]
 51%|█████▏    | 20/39 [14:20<13:37, 43.04s/it]
 54%|█████▍    | 21/39 [14:58<12:27, 41.54s/it]
 56%|█████▋    | 22/39 [15:43<12:01, 42.45s/it]
 59%|█████▉    | 23/39 [16:29<11:35, 43.49s/it]
 62%|██████▏   | 24/39 [17:13<10:55, 43.69s/it]
 64%|██████▍   | 25/39 [18:02<10:34, 45.29s/it]
 67%|██████▋   | 26/39 [18:41<09:25, 43.53s/it]
 69%|██████▉   | 27/39 [19:26<08:45, 43.77s/it]
 72%|███████▏  | 28/39 [20:04<07:44, 42.22s/it]
 74%|███████▍  | 29/39 [20:45<06:56, 41.65s/it]
 77%|███████▋  | 30/39 [21:32<06:30, 43.44s/it]
 79%|███████▉  | 31/39 [22:16<05:47, 43.46s/it]
 82%|████████▏ | 32/39 [22:52<04:49, 41.32s/it]
 85%|████████▍ | 33/39 [23:36<04:13, 42.19s/it]
 87%|████████▋ | 34/39 [24:18<03:29, 41.99s/it]
 90%|████████▉ | 35/39 [25:00<02:48, 42.17s/it]
 92%|█████████▏| 36/39 [25:46<02:09, 43.24s/it]
 95%|█████████▍| 37/39 [26:29<01:26, 43.24s/it]
 97%|█████████▋| 38/39 [27:18<00:44, 44.74s/it]
100%|██████████| 39/39 [27:46<00:00, 42.74s/it]
MAP: 0.43667954206466675
```
### Tensorboard Plots 
1. **Training Loss vs Steps:** 
    ![image](https://github.com/RaviNaik/ERA-SESSION13/assets/23289802/5cb753e0-377b-4d9f-a240-871270ed50db) 


2. **Validation Loss vs Steps:** 
    (Info: Validation loss calculated every 10 epochs to save time, thats why the straight line) 
    ![image](https://github.com/RaviNaik/ERA-SESSION13/assets/23289802/7401c0aa-f7ff-4a5b-bab2-dbb5ebe0b400) 



