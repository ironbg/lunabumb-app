# Third-party skills

These project skills are copied unchanged from public repositories. All except the kevinbadi/blender-skills ones are MIT-licensed; the license texts follow.

| Skills | Source | Commit |
|---|---|---|
| threejs-fundamentals, threejs-geometry, threejs-materials, threejs-lighting, threejs-textures, threejs-animation, threejs-loaders, threejs-shaders, threejs-postprocessing, threejs-interaction | https://github.com/cloudai-x/threejs-skills | b1c6230 |
| blender-web-pipeline | https://github.com/freshtechbro/claudedesignskills | 1da73fe |
| threejs-impl-webgpu, threejs-impl-react-three-fiber, threejs-agents-model-optimizer, threejs-errors-performance | https://github.com/OpenAEC-Foundation/Three.js-Claude-Skill-Package | 6c190f0 |
| new-parent-logistics | https://github.com/mohitagw15856/pm-claude-skills | see new-parent-logistics/LICENSE |
| image-to-3d, multi-image-to-3d, turntable, slow-zoom, dolly-rotate, crane-shot, dynamic-full-loop, perfect-loop, product-polish, threejs-export, polyhaven-hdri-showcase, polyhaven-texture-apply, polyhaven-studio-setup, polyhaven-scene-builder, polyhaven-material-swap | https://github.com/kevinbadi/blender-skills | b2f0f81 |

## cloudai-x/threejs-skills

The repository's README states: "MIT License - Feel free to use, modify, and distribute." It ships no separate LICENSE file.

## freshtechbro/claudedesignskills

```
MIT License

Copyright (c) 2025 Claude Skills Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## OpenAEC-Foundation/Three.js-Claude-Skill-Package

```
MIT License

Copyright (c) 2026 OpenAEC Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## kevinbadi/blender-skills

The repository has no license file and its README names no license, so the author keeps all rights. Check with the author before sharing these skills outside this project.

The repository's `blender-toolkit` skill was not copied: it needs a config file and scripts from a separate plugin that the repository does not include.

These skills drive a running Blender through the [blender-mcp](https://github.com/ahujasid/blender-mcp) addon on `localhost:9876`. The image-to-3D skills also need a Meshy API key (`MESHY_API_KEY`), and the video skills need ffmpeg. Some examples use the author's macOS paths (`/Users/kevinbahrabadi/...`, `~/Desktop/Blender Videos/`); change them to a local folder before use.
