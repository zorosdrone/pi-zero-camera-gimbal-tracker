# Third-party notices and scope

## Picamera2-derived examples

The following files cite or adapt Raspberry Pi Picamera2 example material:

- `src/01A_camera_still_test.py`
- `src/01B_camera_dual_stream_test.py`
- `src/02A_mjpeg_stream_test.py`

They retain the required BSD 2-Clause notice below.

```text
BSD 2-Clause License

Copyright (c) 2021, Raspberry Pi
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Dependencies

The repository does not bundle Picamera2, OpenCV, PyAV, Edge Impulse SDK,
`gpiozero`, `pigpio`, or their transitive dependencies. Obtain them from their
official distribution channels and comply with their respective licenses.

## SG90 2-axis gimbal model

The physical project uses the external MakerWorld model [SG90 Servo 2 Axis
Gimbal](https://makerworld.com/ja/models/511916-sg90-servo-2-axis-gimbal).
No CAD/model file or derivative model is included here, and this repository
does not grant any license for it. Before publishing or redistributing that
model, verify its author, license, and terms on the source page.

## Exclusions

Pre-trained/inference models, datasets, product-page images, and person sample
images are not included and are outside the licenses in this repository.
