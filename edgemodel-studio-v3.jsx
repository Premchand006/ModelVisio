import { useState, useRef, useCallback, useEffect, createContext, useContext } from "react";

const Ctx = createContext();
const useT = () => useContext(Ctx);

const TH = {
  dark: {
    bg:"#05070E",bg1:"#0B1120",bg2:"#111827",bg3:"#1F2937",
    bdr:"#1E293B",bdr2:"#334155",
    t0:"#F0F6FC",t1:"#CBD5E1",t2:"#64748B",t3:"#475569",
    acc:"#38BDF8",acc2:"#818CF8",acc3:"#2DD4BF",
    suc:"#34D399",wrn:"#FBBF24",err:"#F87171",
    glow:"rgba(56,189,248,0.15)",
    cI:"#6EE7B7",cB:"#60A5FA",cN:"#C084FC",cH:"#F472B6",cO:"#FBBF24",
    nodeGrad1:"#0F172A",nodeGrad2:"#1E293B",
    glass:"rgba(15,23,42,0.7)",glassBdr:"rgba(56,189,248,0.2)",
  },
  light: {
    bg:"#F1F5F9",bg1:"#FFFFFF",bg2:"#F8FAFC",bg3:"#E2E8F0",
    bdr:"#CBD5E1",bdr2:"#94A3B8",
    t0:"#0F172A",t1:"#334155",t2:"#64748B",t3:"#94A3B8",
    acc:"#0284C7",acc2:"#6366F1",acc3:"#0D9488",
    suc:"#059669",wrn:"#D97706",err:"#DC2626",
    glow:"rgba(2,132,199,0.08)",
    cI:"#059669",cB:"#2563EB",cN:"#7C3AED",cH:"#DB2777",cO:"#D97706",
    nodeGrad1:"#FFFFFF",nodeGrad2:"#F1F5F9",
    glass:"rgba(255,255,255,0.8)",glassBdr:"rgba(2,132,199,0.15)",
  }
};

const fmt=(n)=>{if(n>=1e9)return(n/1e9).toFixed(2)+"G";if(n>=1e6)return(n/1e6).toFixed(1)+"M";if(n>=1e3)return(n/1e3).toFixed(1)+"K";return""+n};
const fB=(b)=>{if(b>=1e6)return(b/1e6).toFixed(1)+"MB";if(b>=1e3)return(b/1e3).toFixed(1)+"KB";return b+"B"};

// ═══════════ YOLO26n MODEL ═══════════
const DEMO = {
  name:"yolo26n.onnx",format:"ONNX",framework:"Ultralytics YOLO26",opset:18,irVersion:9,
  producer:"pytorch 2.4 + ultralytics 8.4",sizeBytes:5_283_840,
  inputShape:[1,3,640,640],outputShape:[1,84,8400],
  doc:"YOLO26n — NMS-free end-to-end nano detector. DFL removed, dual-head (one2one inference).",
  layers:[
    {id:0,name:"images",type:"Input",op:"Placeholder",shape:"1×3×640×640",dt:"float32",params:0,flops:0,macs:0,mem:4915200,group:"input",
      ins:[],outs:[{n:"images",s:[1,3,640,640]}],attr:{},w:null,
      math:"Input: N×C×H×W = 1×3×640×640\nMemory: 4.69 MB\nPreprocess: /255.0, BGR→RGB",
      insight:"Entry point. YOLO26 keeps 640×640 default. NMS-free means output is directly usable — no post-processing needed.",
      qSens:0,compIssues:[]},
    {id:1,name:"stem.conv",type:"Conv2d",op:"Conv",shape:"1×16×320×320",dt:"float32",params:432,flops:88_473_600,macs:44_236_800,mem:6553600,group:"backbone",
      ins:[{n:"images",s:[1,3,640,640]}],outs:[{n:"stem_out",s:[1,16,320,320]}],
      attr:{kernel:[3,3],stride:[2,2],pad:[1,1,1,1],dilation:[1,1],groups:1},
      w:{shape:[16,3,3,3],size:1728,min:-0.31,max:0.30,mean:0.002,std:0.14,sparse:0},
      math:"Y = Conv2d(X, W, b)\nOut = ⌊(640+2−3)/2⌋+1 = 320\nFLOPs = 2×16×3×9×320² = 88.5M",
      insight:"Stem conv — stride-2 halves resolution immediately. Only 432 params but processes largest feature map. Memory-bound on edge.",
      qSens:0.02,compIssues:[]},
    {id:2,name:"stem.act",type:"SiLU",op:"Mul(Sigmoid)",shape:"1×16×320×320",dt:"float32",params:0,flops:3_276_800,macs:1_638_400,mem:6553600,group:"backbone",
      ins:[{n:"stem_out",s:[1,16,320,320]}],outs:[{n:"stem_act",s:[1,16,320,320]}],attr:{},w:null,
      math:"SiLU(x) = x · σ(x) = x/(1+e^(−x))\nSwish activation, decomposes to Sigmoid+Mul",
      insight:"SiLU not natively supported on some NPUs. Decomposes to Sigmoid+Mul (2 ops). Consider HardSwish on Coral/RK3588.",
      qSens:0.01,compIssues:[{target:"coral_tpu",severity:"warn",msg:"SiLU decomposes to Sigmoid+Mul — 2 TPU ops instead of 1"},{target:"rk3588",severity:"warn",msg:"RKNN v1.x uses CPU fallback for SiLU. v2.0+ supports natively."}]},
    {id:3,name:"conv1.conv",type:"Conv2d",op:"Conv",shape:"1×32×160×160",dt:"float32",params:4608,flops:117_964_800,macs:58_982_400,mem:3276800,group:"backbone",
      ins:[{n:"stem_act",s:[1,16,320,320]}],outs:[{n:"conv1_out",s:[1,32,160,160]}],
      attr:{kernel:[3,3],stride:[2,2],pad:[1,1,1,1],groups:1},
      w:{shape:[32,16,3,3],size:18432,min:-0.25,max:0.23,mean:0.001,std:0.098,sparse:0.02},
      math:"Conv2d: 16→32ch, stride 2\nOut: 160×160\nFLOPs: 118.0M",
      insight:"Second downsample. Feature map area drops 4× but channels double — classic efficiency trade.",
      qSens:0.03,compIssues:[]},
    {id:4,name:"conv1.act",type:"SiLU",op:"Mul(Sigmoid)",shape:"1×32×160×160",dt:"float32",params:0,flops:1_638_400,macs:819_200,mem:3276800,group:"backbone",
      ins:[{n:"conv1_out",s:[1,32,160,160]}],outs:[{n:"conv1_act",s:[1,32,160,160]}],attr:{},w:null,
      math:"SiLU activation",insight:"Fuses with Conv on TensorRT/OpenVINO.",qSens:0.01,compIssues:[]},
    {id:5,name:"c2f_0",type:"C2f",op:"C2f_Block",shape:"1×32×160×160",dt:"float32",params:9280,flops:79_134_720,macs:39_567_360,mem:3276800,group:"backbone",
      ins:[{n:"conv1_act",s:[1,32,160,160]}],outs:[{n:"c2f0_out",s:[1,32,160,160]}],
      attr:{c1:32,c2:32,n:1,shortcut:true,expansion:0.5},
      w:{shape:"[multi]",size:37120,min:-0.29,max:0.28,mean:0,std:0.087,sparse:0.03},
      math:"C2f: Split→1×Bottleneck→Concat→Fuse\nCross-Stage Partial v2 with residual",
      insight:"YOLOv8/26 core block. Cross-stage partial design reuses features efficiently.",
      qSens:0.05,compIssues:[]},
    {id:6,name:"conv2.conv",type:"Conv2d",op:"Conv",shape:"1×64×80×80",dt:"float32",params:18432,flops:117_964_800,macs:58_982_400,mem:1638400,group:"backbone",
      ins:[{n:"c2f0_out",s:[1,32,160,160]}],outs:[{n:"conv2_out",s:[1,64,80,80]}],
      attr:{kernel:[3,3],stride:[2,2],pad:[1,1,1,1],groups:1},
      w:{shape:[64,32,3,3],size:73728,min:-0.20,max:0.19,mean:0,std:0.072,sparse:0.04},
      math:"Conv2d: 32→64, stride 2 → 80×80 (P3)",
      insight:"P3 level — primary small-object detection resolution.",qSens:0.04,compIssues:[]},
    {id:7,name:"conv2.act",type:"SiLU",op:"Mul(Sigmoid)",shape:"1×64×80×80",dt:"float32",params:0,flops:1_638_400,macs:819_200,mem:1638400,group:"backbone",
      ins:[{n:"conv2_out",s:[1,64,80,80]}],outs:[{n:"conv2_act",s:[1,64,80,80]}],attr:{},w:null,
      math:"SiLU activation",insight:"Fuses in optimized runtimes.",qSens:0.01,compIssues:[]},
    {id:8,name:"c2f_1",type:"C2f",op:"C2f_Block",shape:"1×64×80×80",dt:"float32",params:49664,flops:126_689_280,macs:63_344_640,mem:1638400,group:"backbone",
      ins:[{n:"conv2_act",s:[1,64,80,80]}],outs:[{n:"c2f1_out",s:[1,64,80,80]}],
      attr:{c1:64,c2:64,n:2,shortcut:true,expansion:0.5},
      w:{shape:"[multi]",size:198656,min:-0.18,max:0.18,mean:0,std:0.063,sparse:0.05},
      math:"C2f: 64ch, n=2 bottlenecks\nParams: 49,664 | FLOPs: 126.7M",
      insight:"Rich feature extraction. Key for quantization — INT8 can lose fine-grained features here.",
      qSens:0.12,compIssues:[]},
    {id:9,name:"conv3.conv",type:"Conv2d",op:"Conv",shape:"1×128×40×40",dt:"float32",params:73728,flops:117_964_800,macs:58_982_400,mem:819200,group:"backbone",
      ins:[{n:"c2f1_out",s:[1,64,80,80]}],outs:[{n:"conv3_out",s:[1,128,40,40]}],
      attr:{kernel:[3,3],stride:[2,2],pad:[1,1,1,1],groups:1},
      w:{shape:[128,64,3,3],size:294912},
      math:"Conv2d: 64→128, stride 2 → 40×40 (P4)",
      insight:"P4 feature level — medium-to-large objects.",qSens:0.06,compIssues:[]},
    {id:10,name:"conv3.act",type:"SiLU",op:"Mul(Sigmoid)",shape:"1×128×40×40",dt:"float32",params:0,flops:819_200,macs:409_600,mem:819200,group:"backbone",
      ins:[{n:"conv3_out",s:[1,128,40,40]}],outs:[{n:"conv3_act",s:[1,128,40,40]}],attr:{},w:null,
      math:"SiLU",insight:"Fuses with conv.",qSens:0.01,compIssues:[]},
    {id:11,name:"c2f_2",type:"C2f",op:"C2f_Block",shape:"1×128×40×40",dt:"float32",params:197632,flops:126_361_600,macs:63_180_800,mem:819200,group:"backbone",
      ins:[{n:"conv3_act",s:[1,128,40,40]}],outs:[{n:"c2f2_out",s:[1,128,40,40]}],
      attr:{c1:128,c2:128,n:2,shortcut:true,expansion:0.5},
      w:{shape:"[multi]",size:790528},
      math:"C2f: 128ch, n=2 bottlenecks\n12.5% of model params",
      insight:"Contains 12.5% of all parameters. Prime candidate for pruning.",
      qSens:0.15,compIssues:[]},
    {id:12,name:"conv4.conv",type:"Conv2d",op:"Conv",shape:"1×256×20×20",dt:"float32",params:294912,flops:117_964_800,macs:58_982_400,mem:409600,group:"backbone",
      ins:[{n:"c2f2_out",s:[1,128,40,40]}],outs:[{n:"conv4_out",s:[1,256,20,20]}],
      attr:{kernel:[3,3],stride:[2,2],pad:[1,1,1,1],groups:1},
      w:{shape:[256,128,3,3],size:1179648},
      math:"Conv2d: 128→256, stride 2 → 20×20 (P5)\nWeights: 1.12 MB — largest tensor",
      insight:"⚠ Weight tensor >1MB. May need tiling on NPUs with small SRAM. P5 captures large objects.",
      qSens:0.08,compIssues:[{target:"coral_tpu",severity:"error",msg:"Weight tensor 1.12MB exceeds 64MB SRAM budget when combined with activations"}]},
    {id:13,name:"conv4.act",type:"SiLU",op:"Mul(Sigmoid)",shape:"1×256×20×20",dt:"float32",params:0,flops:409_600,macs:204_800,mem:409600,group:"backbone",
      ins:[{n:"conv4_out",s:[1,256,20,20]}],outs:[{n:"conv4_act",s:[1,256,20,20]}],attr:{},w:null,
      math:"SiLU",insight:"Fuses with conv.",qSens:0.01,compIssues:[]},
    {id:14,name:"c2f_3",type:"C2f",op:"C2f_Block",shape:"1×256×20×20",dt:"float32",params:460288,flops:73_646_080,macs:36_823_040,mem:409600,group:"backbone",
      ins:[{n:"conv4_act",s:[1,256,20,20]}],outs:[{n:"c2f3_out",s:[1,256,20,20]}],
      attr:{c1:256,c2:256,n:1,shortcut:true,expansion:0.5},
      w:{shape:"[multi]",size:1841152},
      math:"C2f: 256ch, n=1\n⚠ 29% of all parameters on 20×20 map\nHigh param density, low spatial resolution",
      insight:"⚠ BOTTLENECK: 29% of params in one block. Prime target for INT8 — accuracy loss usually minimal on low-res features.",
      qSens:0.08,compIssues:[]},
    {id:15,name:"sppf",type:"SPPF",op:"MaxPool×3+Concat+Conv",shape:"1×256×20×20",dt:"float32",params:164608,flops:26_214_400,macs:13_107_200,mem:1228800,group:"backbone",
      ins:[{n:"c2f3_out",s:[1,256,20,20]}],outs:[{n:"sppf_out",s:[1,256,20,20]}],
      attr:{k:5,pools:3,out_ch:256},
      w:{shape:"Conv 1×1: 128→256",size:658432},
      math:"SPPF:\n1. Conv 1×1: 256→128\n2. MaxPool(k=5,s=1,p=2) ×3 cascaded\n3. Concat: [orig,p1,p2,p3] = 512ch\n4. Conv 1×1: 512→256\n\nEffective RF: 5+5+5 = 13",
      insight:"Backbone terminus. Triple MaxPool chain = 13×13 receptive field. Memory spike: 3× intermediates. Coral TPU compiles MaxPool chains poorly.",
      qSens:0.1,compIssues:[{target:"coral_tpu",severity:"error",msg:"Cascaded MaxPool not supported. Requires restructuring to parallel pools."},{target:"kneron_kl720",severity:"error",msg:"SPPF exceeds KL720 memory budget."}]},
    {id:16,name:"upsample0",type:"Upsample",op:"Resize",shape:"1×256×40×40",dt:"float32",params:0,flops:0,macs:0,mem:1638400,group:"neck",
      ins:[{n:"sppf_out",s:[1,256,20,20]}],outs:[{n:"up0",s:[1,256,40,40]}],
      attr:{mode:"nearest",scale:2},w:null,
      math:"Nearest upsample: 20×20 → 40×40\nZero FLOPs — memory reshape",
      insight:"Free compute but allocates new memory buffer.",
      qSens:0,compIssues:[{target:"coral_tpu",severity:"warn",msg:"Resize op falls back to CPU on EdgeTPU."},{target:"rk3588",severity:"warn",msg:"Nearest resize may use CPU path on RKNN v1.x."}]},
    {id:17,name:"concat0",type:"Concat",op:"Concat",shape:"1×384×40×40",dt:"float32",params:0,flops:0,macs:0,mem:2457600,group:"neck",
      ins:[{n:"up0",s:[1,256,40,40]},{n:"c2f2_out",s:[1,128,40,40]}],
      outs:[{n:"cat0",s:[1,384,40,40]}],
      attr:{axis:1},w:null,
      math:"Concat dim=1: [256,40,40]⊕[128,40,40] → [384,40,40]\nSkip from backbone P4",
      insight:"FPN skip connection. Peak memory: all three tensors coexist = 4.68 MB. Often the memory high-water mark.",
      qSens:0,compIssues:[]},
    {id:18,name:"c2f_4 (neck)",type:"C2f",op:"C2f_Block",shape:"1×128×40×40",dt:"float32",params:115712,flops:73_891_840,macs:36_945_920,mem:819200,group:"neck",
      ins:[{n:"cat0",s:[1,384,40,40]}],outs:[{n:"c2f4_out",s:[1,128,40,40]}],
      attr:{c1:384,c2:128,n:1,shortcut:false},
      w:{shape:"[multi]",size:462848},
      math:"C2f: 384→128, no shortcut (channel mismatch)\nNeck fusion block",
      insight:"Reduces concatenated channels. No residual since in≠out channels.",
      qSens:0.07,compIssues:[]},
    {id:19,name:"upsample1",type:"Upsample",op:"Resize",shape:"1×128×80×80",dt:"float32",params:0,flops:0,macs:0,mem:3276800,group:"neck",
      ins:[{n:"c2f4_out",s:[1,128,40,40]}],outs:[{n:"up1",s:[1,128,80,80]}],
      attr:{mode:"nearest",scale:2},w:null,
      math:"Nearest: 40×40 → 80×80",insight:"Second upsample for P3 fusion.",
      qSens:0,compIssues:[]},
    {id:20,name:"concat1",type:"Concat",op:"Concat",shape:"1×192×80×80",dt:"float32",params:0,flops:0,macs:0,mem:4915200,group:"neck",
      ins:[{n:"up1",s:[1,128,80,80]},{n:"c2f1_out",s:[1,64,80,80]}],
      outs:[{n:"cat1",s:[1,192,80,80]}],
      attr:{axis:1},w:null,
      math:"Concat: [128,80,80]⊕[64,80,80] → [192,80,80]",
      insight:"P3 skip — highest resolution in neck. 192ch×80×80 can stress sub-2GB devices.",
      qSens:0,compIssues:[]},
    {id:21,name:"c2f_5 (neck)",type:"C2f",op:"C2f_Block",shape:"1×64×80×80",dt:"float32",params:29184,flops:74_547_200,macs:37_273_600,mem:1638400,group:"neck",
      ins:[{n:"cat1",s:[1,192,80,80]}],outs:[{n:"c2f5_out",s:[1,64,80,80]}],
      attr:{c1:192,c2:64,n:1,shortcut:false},
      w:{shape:"[multi]",size:116736},
      math:"C2f: 192→64, P3 detection features",
      insight:"P3 output for small-object detection. Critical for surveillance, drones.",
      qSens:0.09,compIssues:[]},
    {id:22,name:"detect.one2one_cv2.0",type:"Conv2d",op:"Conv",shape:"1×64×80×80",dt:"float32",params:36992,flops:75_497_472,macs:37_748_736,mem:1638400,group:"head",
      ins:[{n:"c2f5_out",s:[1,64,80,80]}],outs:[{n:"bbox_p3",s:[1,64,80,80]}],
      attr:{kernel:[3,3],stride:[1,1],pad:[1,1,1,1]},w:{shape:[64,64,3,3],size:147968},
      math:"Detection head — bbox regression (P3)\nYOLO26: NMS-free one2one head\n64 = 4 coords × 16 bins (no DFL in v26!)",
      insight:"YOLO26 removes DFL — simpler, faster bbox regression. One2one head = no NMS needed.",
      qSens:0.06,compIssues:[]},
    {id:23,name:"detect.one2one_cv3.0",type:"Conv2d",op:"Conv",shape:"1×80×80×80",dt:"float32",params:46240,flops:94_371_840,macs:47_185_920,mem:2048000,group:"head",
      ins:[{n:"c2f5_out",s:[1,64,80,80]}],outs:[{n:"cls_p3",s:[1,80,80,80]}],
      attr:{kernel:[3,3],stride:[1,1],pad:[1,1,1,1]},w:{shape:[80,64,3,3],size:184960},
      math:"Classification branch (P3)\n80 = COCO classes",
      insight:"80ch for COCO. Custom datasets with fewer classes shrinks this proportionally.",
      qSens:0.11,compIssues:[]},
    {id:24,name:"detect.one2one_cv2.1",type:"Conv2d",op:"Conv",shape:"1×64×40×40",dt:"float32",params:73856,flops:37_748_736,macs:18_874_368,mem:409600,group:"head",
      ins:[{n:"c2f4_out",s:[1,128,40,40]}],outs:[{n:"bbox_p4",s:[1,64,40,40]}],
      attr:{kernel:[3,3]},w:{shape:[64,128,3,3],size:295424},
      math:"Bbox regression at P4",insight:"P4 bbox head — medium objects.",qSens:0.05,compIssues:[]},
    {id:25,name:"detect.one2one_cv3.1",type:"Conv2d",op:"Conv",shape:"1×80×40×40",dt:"float32",params:92320,flops:47_185_920,macs:23_592_960,mem:512000,group:"head",
      ins:[{n:"c2f4_out",s:[1,128,40,40]}],outs:[{n:"cls_p4",s:[1,80,40,40]}],
      attr:{kernel:[3,3]},w:{shape:[80,128,3,3],size:369280},
      math:"Classification at P4",insight:"P4 class head.",qSens:0.08,compIssues:[]},
    {id:26,name:"detect.one2one_cv2.2",type:"Conv2d",op:"Conv",shape:"1×64×20×20",dt:"float32",params:147584,flops:18_874_368,macs:9_437_184,mem:102400,group:"head",
      ins:[{n:"sppf_out",s:[1,256,20,20]}],outs:[{n:"bbox_p5",s:[1,64,20,20]}],
      attr:{kernel:[3,3]},w:{shape:[64,256,3,3],size:590336},
      math:"Bbox at P5",insight:"P5 bbox — large objects.",qSens:0.04,compIssues:[]},
    {id:27,name:"detect.one2one_cv3.2",type:"Conv2d",op:"Conv",shape:"1×80×20×20",dt:"float32",params:184480,flops:23_592_960,macs:11_796_480,mem:128000,group:"head",
      ins:[{n:"sppf_out",s:[1,256,20,20]}],outs:[{n:"cls_p5",s:[1,80,20,20]}],
      attr:{kernel:[3,3]},w:{shape:[80,256,3,3],size:737280},
      math:"Classification at P5\nHeaviest head layer: 184K params",insight:"P5 class head — heaviest head layer.",qSens:0.07,compIssues:[]},
    {id:28,name:"detect/concat_e2e",type:"Concat",op:"Concat",shape:"1×144×8400",dt:"float32",params:0,flops:0,macs:0,mem:4838400,group:"head",
      ins:[{n:"bbox_p3"},{n:"cls_p3"},{n:"bbox_p4"},{n:"cls_p4"},{n:"bbox_p5"},{n:"cls_p5"}],
      outs:[{n:"raw_e2e",s:[1,144,8400]}],attr:{axis:2},w:null,
      math:"Multi-scale: 80²+40²+20²=8400 candidates\n144 = 64(bbox)+80(cls) per candidate\nYOLO26: End-to-end — NO NMS needed!",
      insight:"NMS-free output. 8400 predictions directly usable. One2one head ensures no duplicate boxes. Massive simplification for edge deploy.",
      qSens:0,compIssues:[]},
    {id:29,name:"output0",type:"Output",op:"Identity",shape:"1×84×8400",dt:"float32",params:0,flops:0,macs:0,mem:2822400,group:"output",
      ins:[{n:"raw_e2e",s:[1,84,8400]}],outs:[{n:"output0",s:[1,84,8400]}],attr:{},w:null,
      math:"Final: [1,84,8400]\n84 = 4(xywh) + 80(classes)\nYOLO26: NMS-free → decode bbox + threshold only\nNo NMS post-processing!",
      insight:"Model endpoint. YOLO26's killer feature: zero post-processing. Just decode boxes and threshold. TensorRT: no EfficientNMS plugin needed.",
      qSens:0,compIssues:[]},
  ],
  edges:[[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],[11,17],[16,17],[17,18],[18,19],[8,20],[19,20],[20,21],[21,22],[21,23],[18,24],[18,25],[15,26],[15,27],[22,28],[23,28],[24,28],[25,28],[26,28],[27,28],[28,29]],
};

const HW=[
  {id:"agx_orin",n:"NVIDIA Jetson AGX Orin",v:"NVIDIA",tops:275,ram:64,w:60,fmts:["ONNX","TensorRT","PyTorch"],sc:99},
  {id:"orin_nx",n:"NVIDIA Jetson Orin NX 16GB",v:"NVIDIA",tops:100,ram:16,w:25,fmts:["ONNX","TensorRT","PyTorch"],sc:97},
  {id:"orin_nano",n:"NVIDIA Jetson Orin Nano",v:"NVIDIA",tops:40,ram:8,w:15,fmts:["ONNX","TensorRT","PyTorch"],sc:94},
  {id:"coral",n:"Google Coral Edge TPU",v:"Google",tops:4,ram:0,w:2,fmts:["TFLite"],sc:58},
  {id:"rb5",n:"Qualcomm RB5",v:"Qualcomm",tops:15,ram:8,w:12,fmts:["ONNX","TFLite","SNPE","QNN"],sc:78},
  {id:"rb6",n:"Qualcomm RB6 (QCS6490)",v:"Qualcomm",tops:20,ram:12,w:15,fmts:["ONNX","TFLite","QNN"],sc:82},
  {id:"intel_npu",n:"Intel Core Ultra NPU",v:"Intel",tops:11,ram:32,w:30,fmts:["ONNX","OpenVINO"],sc:80},
  {id:"movidius",n:"Intel Movidius Myriad X",v:"Intel",tops:4,ram:0.5,w:1.5,fmts:["OpenVINO","ONNX"],sc:52},
  {id:"kv260",n:"AMD Xilinx KV260",v:"AMD/Xilinx",tops:3.2,ram:4,w:35,fmts:["ONNX","Vitis-AI"],sc:55},
  {id:"hailo8",n:"Hailo-8",v:"Hailo",tops:26,ram:0,w:2.5,fmts:["ONNX","TFLite","HailoRT"],sc:85},
  {id:"hailo8l",n:"Hailo-8L (RPi AI Kit)",v:"Hailo",tops:13,ram:0,w:1.5,fmts:["ONNX","TFLite","HailoRT"],sc:72},
  {id:"rk3588",n:"Rockchip RK3588 NPU",v:"Rockchip",tops:6,ram:8,w:10,fmts:["ONNX","RKNN","TFLite"],sc:68},
  {id:"sima",n:"SiMa.ai MLSoC",v:"SiMa.ai",tops:50,ram:16,w:10,fmts:["ONNX","TFLite","SiMa"],sc:88},
  {id:"deepx",n:"DeepX DX-M1",v:"DeepX",tops:25,ram:0,w:3,fmts:["ONNX","TFLite","DeepX"],sc:76},
  {id:"mobilint",n:"Mobilint MA35D",v:"Mobilint",tops:30,ram:0,w:5,fmts:["ONNX","Mobilint"],sc:78},
  {id:"kneron",n:"Kneron KL720",v:"Kneron",tops:1.5,ram:0.5,w:1,fmts:["ONNX","TFLite"],sc:48},
  {id:"blaize",n:"Blaize Pathfinder P1600",v:"Blaize",tops:16,ram:8,w:7,fmts:["ONNX","TFLite"],sc:74},
  {id:"nxp",n:"NXP i.MX 93 Neutron NPU",v:"NXP",tops:0.5,ram:2,w:2,fmts:["TFLite","ONNX"],sc:42},
  {id:"amlogic",n:"Amlogic A311D2 NPU",v:"Amlogic",tops:7.2,ram:4,w:8,fmts:["ONNX","TFLite","Tengine"],sc:62},
  {id:"ti_tda4",n:"TI TDA4VM (Jacinto)",v:"TI",tops:8,ram:4,w:20,fmts:["ONNX","TFLite","TIDL"],sc:66},
  {id:"mtk_genio",n:"MediaTek Genio 1200 APU",v:"MediaTek",tops:4.8,ram:8,w:8,fmts:["ONNX","TFLite","NeuroPilot"],sc:64},
];

const FORMATS=["ONNX","TFLite","PyTorch (.pt)","torch.export","ExecuTorch","TorchScript","TensorFlow (SavedModel)","TF.js","Core ML","OpenVINO","Keras (.keras)","Caffe (.caffemodel)","Darknet (.cfg/.weights)","Safetensors","NumPy (.npz)","MLIR","JAX (StableHLO)","GGUF","RKNN","ncnn","MNN","PaddlePaddle","scikit-learn (.pkl)","TensorRT (.engine)"];

const CONV_SCRIPTS = {
  "pt→onnx":(o)=>`from ultralytics import YOLO\nmodel = YOLO("${o.inp||'yolo26n.pt'}")\nmodel.export(format="onnx", opset=${o.opset||18}, simplify=True, imgsz=${o.sz||640}, dynamic=${o.dyn?"True":"False"})`,
  "pt→tflite":(o)=>`from ultralytics import YOLO\nmodel = YOLO("${o.inp||'yolo26n.pt'}")\nmodel.export(format="tflite", imgsz=${o.sz||640}, int8=${o.i8?"True":"False"})`,
  "onnx→tensorrt":(o)=>`# TensorRT conversion\nimport subprocess\ncmd = ["trtexec", f"--onnx=${o.inp||'model.onnx'}", f"--saveEngine=${o.out||'model.engine'}", "--${o.prec||'fp16'}", "--workspace=4096"${o.dyn?', "--minShapes=images:1x3x640x640", "--optShapes=images:1x3x640x640", "--maxShapes=images:8x3x640x640"':''}]\nsubprocess.run(cmd, check=True)\nprint("✓ TensorRT engine built")`,
  "onnx→openvino":(o)=>`from openvino.tools import mo\nfrom openvino.runtime import serialize\nmodel = mo.convert_model("${o.inp||'model.onnx'}", compress_to_fp16=True, input_shape=[1,3,${o.sz||640},${o.sz||640}])\nserialize(model, "${o.out||'model.xml'}")\nprint("✓ OpenVINO IR saved")`,
  "onnx→rknn":(o)=>`from rknn.api import RKNN\nrknn = RKNN()\nrknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform="${o.plat||'rk3588'}")\nrknn.load_onnx(model="${o.inp||'model.onnx'}")\nrknn.build(do_quantization=${o.i8?"True":"False"}${o.i8?', dataset="calib_list.txt"':''})\nrknn.export_rknn("${o.out||'model.rknn'}")\nprint("✓ RKNN exported")`,
  "onnx→ncnn":(o)=>`# ONNX → ncnn\nimport subprocess\nsubprocess.run(["onnx2ncnn", "${o.inp||'model.onnx'}", "${o.out||'model'}.param", "${o.out||'model'}.bin"], check=True)\n# Optimize\nsubprocess.run(["ncnnoptimize", "${o.out||'model'}.param", "${o.out||'model'}.bin", "${o.out||'model'}_opt.param", "${o.out||'model'}_opt.bin", "${o.prec==='fp16'?'1':'0'}"], check=True)\nprint("✓ ncnn model exported")`,
  "pt→coreml":(o)=>`from ultralytics import YOLO\nmodel = YOLO("${o.inp||'yolo26n.pt'}")\nmodel.export(format="coreml", imgsz=${o.sz||640}, half=${o.fp16?"True":"False"}, nms=True)\nprint("✓ CoreML .mlpackage exported")`,
};

// ═══════════ COMPONENTS ═══════════

function Badge({children,c}){const t=useT();return<span style={{display:"inline-block",padding:"1px 6px",borderRadius:3,fontSize:9,fontWeight:700,background:(c||t.acc)+"18",color:c||t.acc,border:`1px solid ${(c||t.acc)}30`,fontFamily:"'JetBrains Mono',monospace"}}>{children}</span>}

function Tabs({tabs,active,onChange}){const t=useT();return<div style={{display:"flex",gap:1,padding:3,background:t.bg+"88",borderRadius:6,backdropFilter:"blur(8px)"}}>
  {tabs.map(tb=><button key={tb.id} onClick={()=>onChange(tb.id)} style={{padding:"5px 10px",borderRadius:4,border:"none",fontSize:10,fontWeight:500,cursor:"pointer",transition:"all .15s",background:active===tb.id?t.bg2:"transparent",color:active===tb.id?t.t0:t.t2}}>{tb.icon&&<span style={{marginRight:3}}>{tb.icon}</span>}{tb.label}</button>)}
</div>}

const GC={input:"cI",backbone:"cB",neck:"cN",head:"cH",output:"cO"};

// ═══════════ DAG GRAPH (Netron-style) ═══════════
function DAGGraph({model,selected,onSelect,showMath,qHeatmap}){
  const t=useT(); const ref=useRef(); if(!model) return null;
  const L=model.layers, E=model.edges;
  const nW=190,nH=showMath?72:50,gY=showMath?10:8;

  // Compute ranks via longest path
  const rank={};L.forEach(l=>rank[l.id]=0);
  for(let i=0;i<L.length;i++) E.forEach(([u,v])=>{if(rank[v]<=rank[u]) rank[v]=rank[u]+1;});

  // Group by rank
  const byRank={};L.forEach(l=>{const r=rank[l.id];if(!byRank[r])byRank[r]=[];byRank[r].push(l);});
  const maxRank=Math.max(...Object.keys(byRank).map(Number));

  // Position nodes
  const pos={};
  Object.entries(byRank).forEach(([r,nodes])=>{
    const rn=Number(r);
    const totalW=nodes.length*nW+(nodes.length-1)*24;
    const startX=(Math.max(totalW,nW+40)-totalW)/2+20;
    nodes.forEach((n,i)=>{pos[n.id]={x:startX+i*(nW+24),y:rn*(nH+gY)+16};});
  });

  const svgW=Math.max(...L.map(l=>pos[l.id].x))+nW+40;
  const svgH=(maxRank+1)*(nH+gY)+40;

  // Op type colors
  const opCol=(type)=>{
    if(type.includes("Conv"))return{bg:"#1E3A5F",hd:"#2563EB",tx:"#93C5FD"};
    if(type==="SiLU")return{bg:"#4A1D1D",hd:"#DC2626",tx:"#FCA5A5"};
    if(type==="C2f")return{bg:"#1E3A5F",hd:"#3B82F6",tx:"#BFDBFE"};
    if(type==="SPPF")return{bg:"#2D1F4E",hd:"#7C3AED",tx:"#C4B5FD"};
    if(type==="Concat")return{bg:"#1A3328",hd:"#059669",tx:"#A7F3D0"};
    if(type==="Upsample"||type==="Resize")return{bg:"#2D2A14",hd:"#CA8A04",tx:"#FDE68A"};
    if(type==="MaxPool")return{bg:"#1A3328",hd:"#10B981",tx:"#A7F3D0"};
    if(type==="Input"||type==="Output")return{bg:"#1F2937",hd:"#6B7280",tx:"#D1D5DB"};
    return{bg:"#1F2937",hd:"#6B7280",tx:"#9CA3AF"};
  };

  return(
    <div ref={ref} style={{flex:1,overflow:"auto",background:`radial-gradient(circle at 50% 30%,${t.bg1},${t.bg})`,position:"relative"}}>
      {/* Grid dots */}
      <div style={{position:"absolute",inset:0,backgroundImage:`radial-gradient(${t.bdr} 1px, transparent 1px)`,backgroundSize:"20px 20px",opacity:.3,pointerEvents:"none"}}/>
      <svg width={svgW} height={svgH} style={{display:"block",margin:"0 auto",position:"relative"}}>
        <defs>
          <marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0,8 3,0 6" fill={t.t3}/></marker>
          <filter id="glow"><feGaussianBlur stdDeviation="3" result="g"/><feMerge><feMergeNode in="g"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          <filter id="nodeGlow"><feDropShadow dx="0" dy="0" stdDeviation="6" floodColor={t.acc} floodOpacity="0.4"/></filter>
        </defs>

        {/* Edges */}
        {E.map(([from,to],i)=>{
          const p1=pos[from],p2=pos[to];if(!p1||!p2) return null;
          const x1=p1.x+nW/2,y1_=p1.y+nH,x2=p2.x+nW/2,y2_=p2.y;
          const dx=Math.abs(x2-x1);
          // Edge label (tensor shape)
          const fromL=L.find(l=>l.id===from);
          const shapeLabel=fromL?fromL.shape.split("×").slice(0,4).join("×"):"";
          if(dx<2){
            const midY=(y1_+y2_)/2;
            return<g key={i}>
              <line x1={x1} y1={y1_} x2={x2} y2={y2_} stroke={t.bdr} strokeWidth={1.5} markerEnd="url(#arr)"/>
              {showMath&&<text x={x1+4} y={midY} fontSize={7.5} fill={t.t3} fontFamily="'JetBrains Mono',monospace">{shapeLabel}</text>}
            </g>;
          }
          const cy1=y1_+(y2_-y1_)*0.3,cy2=y1_+(y2_-y1_)*0.7;
          return<g key={i}>
            <path d={`M${x1},${y1_} C${x1},${cy1} ${x2},${cy2} ${x2},${y2_}`} fill="none" stroke={t.bdr} strokeWidth={1.5} markerEnd="url(#arr)"/>
            {showMath&&<text x={(x1+x2)/2+4} y={(y1_+y2_)/2} fontSize={7.5} fill={t.t3} fontFamily="'JetBrains Mono',monospace">{shapeLabel}</text>}
          </g>;
        })}

        {/* Nodes */}
        {L.map(l=>{
          const p=pos[l.id];const sel=selected?.id===l.id;
          const gc=t[GC[l.group]]||t.t3;
          const oc=opCol(l.type);
          const qColor=qHeatmap&&l.qSens>0?(l.qSens>0.12?"#EF4444":l.qSens>0.06?"#F59E0B":"#22C55E"):null;
          const hasIssues=l.compIssues?.length>0;

          return<g key={l.id} onClick={()=>onSelect(l)} style={{cursor:"pointer"}} filter={sel?"url(#nodeGlow)":undefined}>
            {/* Node body */}
            <rect x={p.x} y={p.y} width={nW} height={nH} rx={8}
              fill={qColor?qColor+"15":oc.bg} stroke={sel?t.acc:qColor||gc+"66"} strokeWidth={sel?2:1}
              style={{transition:"all .2s"}}/>
            {/* Header bar */}
            <rect x={p.x} y={p.y} width={nW} height={20} rx={8} fill={oc.hd+"CC"}/>
            <rect x={p.x} y={p.y+12} width={nW} height={8} fill={oc.hd+"CC"}/>
            {/* Type label */}
            <text x={p.x+8} y={p.y+14} fontSize={11} fontWeight={700} fill="#FFF" fontFamily="'JetBrains Mono',monospace">{l.type}</text>
            {/* Issue indicator */}
            {hasIssues&&<circle cx={p.x+nW-10} cy={p.y+10} r={4} fill={t.err}/>}
            {/* Quant sensitivity dot */}
            {qColor&&<circle cx={p.x+nW-22} cy={p.y+10} r={4} fill={qColor}/>}
            {/* Weight info */}
            {l.w&&<text x={p.x+8} y={p.y+33} fontSize={8.5} fill={t.t2} fontFamily="'JetBrains Mono',monospace">
              W ({typeof l.w.shape==="string"?l.w.shape:l.w.shape.join("×")})
            </text>}
            {!l.w&&<text x={p.x+8} y={p.y+33} fontSize={8.5} fill={t.t3} fontFamily="'JetBrains Mono',monospace">{l.name.length>22?l.name.slice(0,22)+"…":l.name}</text>}
            {/* Shape & params */}
            {showMath&&<>
              <text x={p.x+8} y={p.y+45} fontSize={8} fill={t.acc} fontFamily="'JetBrains Mono',monospace">{l.shape}</text>
              {l.params>0&&<text x={p.x+nW-8} y={p.y+45} fontSize={8} fill={t.t3} textAnchor="end" fontFamily="'JetBrains Mono',monospace">{fmt(l.params)}p</text>}
              {l.flops>0&&<text x={p.x+8} y={p.y+57} fontSize={7.5} fill={t.acc2} fontFamily="'JetBrains Mono',monospace">{fmt(l.flops)} FLOPs</text>}
              {l.macs>0&&<text x={p.x+nW-8} y={p.y+57} fontSize={7.5} fill={t.t3} textAnchor="end" fontFamily="'JetBrains Mono',monospace">{fmt(l.macs)} MACs</text>}
            </>}
            {/* Group color strip */}
            <rect x={p.x} y={p.y} width={3} height={nH} rx={1.5} fill={gc}/>
          </g>;
        })}
      </svg>
    </div>
  );
}

// ═══════════ LAYER INSPECTOR ═══════════
function Inspector({layer,model}){
  const t=useT();const[tab,setTab]=useState("props");
  if(!layer) return<div style={{padding:24,textAlign:"center",color:t.t3,fontSize:11}}>Select a layer node to inspect</div>;
  const gc=t[GC[layer.group]]||t.t3;
  const totalP=model.layers.reduce((s,l)=>s+l.params,0);
  const pct=totalP>0?((layer.params/totalP)*100).toFixed(1):0;
  const R=({l,v,c})=><div style={{display:"flex",justifyContent:"space-between",padding:"4px 0",borderBottom:`1px solid ${t.bdr}22`}}>
    <span style={{fontSize:10,color:t.t2}}>{l}</span>
    <span style={{fontSize:10,color:c||t.t1,fontFamily:"'JetBrains Mono',monospace",textAlign:"right",maxWidth:"60%",wordBreak:"break-all"}}>{v}</span>
  </div>;
  const TABS=[{id:"props",label:"Properties",icon:"⊞"},{id:"tensors",label:"Tensors",icon:"◫"},{id:"math",label:"Math",icon:"∑"},{id:"metrics",label:"Metrics",icon:"◉"},{id:"ai",label:"AI",icon:"✦"}];
  return<div style={{display:"flex",flexDirection:"column",gap:6,height:"100%"}}>
    <div style={{display:"flex",alignItems:"center",gap:6,flexWrap:"wrap"}}>
      <Badge c={gc}>{layer.group}</Badge><span style={{fontSize:13,fontWeight:700,color:t.t0}}>{layer.name}</span><Badge c={t.t3}>{layer.op}</Badge>
    </div>
    <Tabs tabs={TABS} active={tab} onChange={setTab}/>
    <div style={{flex:1,overflow:"auto",background:t.glass,backdropFilter:"blur(12px)",borderRadius:8,border:`1px solid ${t.glassBdr}`,padding:10}}>
      {tab==="props"&&<div>
        <R l="Name" v={layer.name}/><R l="Type" v={layer.type}/><R l="Op" v={layer.op}/><R l="Shape" v={layer.shape} c={t.acc}/><R l="Dtype" v={layer.dt||"float32"}/><R l="Group" v={layer.group}/>
        {Object.keys(layer.attr||{}).length>0&&<><div style={{fontSize:9,fontWeight:700,color:t.t3,textTransform:"uppercase",letterSpacing:1,marginTop:8,marginBottom:4}}>Attributes</div>
        {Object.entries(layer.attr).map(([k,v])=><R key={k} l={k} v={JSON.stringify(v)} c={t.acc}/>)}</>}
      </div>}
      {tab==="tensors"&&<div>
        <div style={{fontSize:9,fontWeight:700,color:t.t3,textTransform:"uppercase",letterSpacing:1,marginBottom:4}}>Inputs ({layer.ins?.length||0})</div>
        {(layer.ins||[]).map((inp,i)=><div key={i} style={{background:t.bg2,borderRadius:4,padding:6,marginBottom:4,border:`1px solid ${t.bdr}`}}>
          <div style={{fontSize:10,fontWeight:600,color:t.t1,fontFamily:"'JetBrains Mono',monospace"}}>{inp.n}</div>
          {inp.s&&<div style={{fontSize:9,color:t.acc}}>Shape: {JSON.stringify(inp.s)}</div>}
        </div>)}
        <div style={{fontSize:9,fontWeight:700,color:t.t3,textTransform:"uppercase",letterSpacing:1,marginTop:8,marginBottom:4}}>Outputs ({layer.outs?.length||0})</div>
        {(layer.outs||[]).map((o,i)=><div key={i} style={{background:t.bg2,borderRadius:4,padding:6,marginBottom:4,border:`1px solid ${t.bdr}`}}>
          <div style={{fontSize:10,fontWeight:600,color:t.t1,fontFamily:"'JetBrains Mono',monospace"}}>{o.n}</div>
          {o.s&&<div style={{fontSize:9,color:t.acc}}>Shape: {JSON.stringify(o.s)}</div>}
        </div>)}
        {layer.w&&<><div style={{fontSize:9,fontWeight:700,color:t.t3,textTransform:"uppercase",letterSpacing:1,marginTop:8,marginBottom:4}}>Weights</div>
        <div style={{background:t.bg2,borderRadius:4,padding:6,border:`1px solid ${t.bdr}`}}>
          <R l="Shape" v={typeof layer.w.shape==="object"?JSON.stringify(layer.w.shape):layer.w.shape}/>
          {layer.w.size&&<R l="Elements" v={fmt(layer.w.size)}/>}
          {layer.w.min!=null&&<R l="Range" v={`[${layer.w.min}, ${layer.w.max}]`} c={t.acc}/>}
          {layer.w.mean!=null&&<R l="Mean±Std" v={`${layer.w.mean}±${layer.w.std}`}/>}
          {layer.w.sparse!=null&&<R l="Sparsity" v={`${(layer.w.sparse*100).toFixed(1)}%`}/>}
        </div></>}
      </div>}
      {tab==="math"&&<pre style={{background:t.bg,border:`1px solid ${t.bdr}`,borderRadius:6,padding:10,fontSize:10.5,color:t.acc,fontFamily:"'JetBrains Mono',monospace",lineHeight:1.6,whiteSpace:"pre-wrap",margin:0}}>{layer.math||"No math info."}</pre>}
      {tab==="metrics"&&<div>
        <R l="Parameters" v={`${fmt(layer.params)} (${pct}%)`}/><R l="FLOPs" v={fmt(layer.flops)} c={t.acc}/><R l="MACs" v={fmt(layer.macs||0)}/><R l="Activation memory" v={fB(layer.mem||0)}/>
        {layer.params>0&&<><R l="Weight mem (FP32)" v={fB(layer.params*4)}/><R l="Weight mem (INT8)" v={fB(layer.params)} c={t.suc}/></>}
        <div style={{fontSize:9,fontWeight:700,color:t.t3,textTransform:"uppercase",letterSpacing:1,marginTop:8,marginBottom:4}}>Est. Layer Latency</div>
        <R l="Jetson Orin Nano (FP16)" v={layer.flops>0?`~${(layer.flops/40e9*1000).toFixed(2)}ms`:"—"}/><R l="Hailo-8 (INT8)" v={layer.flops>0?`~${(layer.flops/26e9*1000).toFixed(2)}ms`:"—"}/><R l="Coral TPU (INT8)" v={layer.flops>0?`~${(layer.flops/4e9*1000).toFixed(2)}ms`:"—"}/>
        <div style={{fontSize:9,fontWeight:700,color:t.t3,textTransform:"uppercase",letterSpacing:1,marginTop:8,marginBottom:4}}>Quantization Sensitivity</div>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <div style={{flex:1,height:6,background:t.bg3,borderRadius:3,overflow:"hidden"}}><div style={{width:`${(layer.qSens||0)*100/0.2*100}%`,maxWidth:"100%",height:"100%",background:layer.qSens>0.12?t.err:layer.qSens>0.06?t.wrn:t.suc,borderRadius:3}}/></div>
          <span style={{fontSize:10,fontWeight:700,color:layer.qSens>0.12?t.err:layer.qSens>0.06?t.wrn:t.suc,fontFamily:"'JetBrains Mono',monospace"}}>{((layer.qSens||0)*100).toFixed(0)}%</span>
        </div>
      </div>}
      {tab==="ai"&&<div style={{background:t.acc2+"12",border:`1px solid ${t.acc2}33`,borderRadius:6,padding:10}}>
        <div style={{fontSize:9,fontWeight:700,color:t.acc2,textTransform:"uppercase",letterSpacing:1,marginBottom:4}}>✦ AI Analysis</div>
        <div style={{fontSize:11,color:t.t1,lineHeight:1.6}}>{layer.insight||"No insight."}</div>
        {layer.compIssues?.length>0&&<><div style={{fontSize:9,fontWeight:700,color:t.err,textTransform:"uppercase",letterSpacing:1,marginTop:10,marginBottom:4}}>⚠ Compiler Issues</div>
        {layer.compIssues.map((ci,i)=><div key={i} style={{background:ci.severity==="error"?t.err+"15":t.wrn+"15",border:`1px solid ${ci.severity==="error"?t.err:t.wrn}33`,borderRadius:4,padding:6,marginBottom:4}}>
          <div style={{fontSize:9,fontWeight:700,color:ci.severity==="error"?t.err:t.wrn}}>{ci.target}</div>
          <div style={{fontSize:10,color:t.t1}}>{ci.msg}</div>
        </div>)}</>}
      </div>}
    </div>
  </div>;
}

// ═══════════ COMPILER CHECKER ═══════════
function CompilerChecker({model}){
  const t=useT();const[target,setTarget]=useState("coral");
  if(!model) return null;
  const allIssues=[];
  model.layers.forEach(l=>{(l.compIssues||[]).forEach(ci=>{allIssues.push({...ci,layer:l.name,layerType:l.type});});});
  const filtered=target==="all"?allIssues:allIssues.filter(ci=>ci.target.includes(target));
  const targets=[{id:"all",label:"All"},{id:"coral",label:"Coral TPU"},{id:"rk3588",label:"RK3588"},{id:"kneron",label:"Kneron"},{id:"hailo",label:"Hailo"}];
  const errors=filtered.filter(i=>i.severity==="error").length;
  const warns=filtered.filter(i=>i.severity==="warn").length;

  // Auto-fix suggestions
  const fixes=[
    {issue:"SiLU decomposition",fix:"Replace SiLU → HardSwish",impact:"Accuracy: -0.2% | Latency: +15% faster",layers:model.layers.filter(l=>l.type==="SiLU").map(l=>l.name)},
    {issue:"Cascaded MaxPool (SPPF)",fix:"Replace SPPF → Parallel SPP",impact:"Coral compatible | Latency: +5%",layers:["sppf"]},
    {issue:"Resize op CPU fallback",fix:"Replace Nearest → TransposedConv",impact:"Fully on-chip | +2K params",layers:model.layers.filter(l=>l.type==="Upsample").map(l=>l.name)},
    {issue:"Large weight tensors (>1MB)",fix:"Apply channel pruning (0.75×)",impact:"Params: -25% | mAP: -0.5%",layers:model.layers.filter(l=>l.params>200000).map(l=>l.name)},
  ];

  return<div style={{display:"flex",flexDirection:"column",gap:8,height:"100%"}}>
    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
      <div style={{fontSize:13,fontWeight:700,color:t.t0}}>Compiler Compatibility Pre-Flight</div>
      <div style={{display:"flex",gap:6}}>
        <Badge c={t.err}>{errors} errors</Badge><Badge c={t.wrn}>{warns} warnings</Badge>
      </div>
    </div>
    <div style={{display:"flex",gap:4,flexWrap:"wrap"}}>
      {targets.map(tg=><button key={tg.id} onClick={()=>setTarget(tg.id)} style={{padding:"3px 8px",borderRadius:4,border:`1px solid ${target===tg.id?t.acc:t.bdr}`,background:target===tg.id?t.acc+"18":"transparent",color:target===tg.id?t.acc:t.t2,fontSize:10,cursor:"pointer"}}>{tg.label}</button>)}
    </div>
    <div style={{flex:1,overflow:"auto",display:"flex",flexDirection:"column",gap:6}}>
      {filtered.length===0&&<div style={{padding:20,textAlign:"center",color:t.suc,fontSize:12}}>✓ No issues detected for this target</div>}
      {filtered.map((ci,i)=><div key={i} style={{background:ci.severity==="error"?t.err+"10":t.wrn+"10",border:`1px solid ${ci.severity==="error"?t.err:t.wrn}30`,borderRadius:6,padding:10}}>
        <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
          <span style={{fontSize:11,fontWeight:600,color:t.t0}}>{ci.layer} <span style={{color:t.t3}}>({ci.layerType})</span></span>
          <Badge c={ci.severity==="error"?t.err:t.wrn}>{ci.severity}</Badge>
        </div>
        <div style={{fontSize:11,color:t.t1}}>{ci.msg}</div>
      </div>)}
      {/* Auto-Fix Engine */}
      <div style={{fontSize:12,fontWeight:700,color:t.acc3,marginTop:8}}>✦ Auto-Fix Suggestions</div>
      {fixes.filter(f=>f.layers.length>0).map((f,i)=><div key={i} style={{background:t.acc3+"08",border:`1px solid ${t.acc3}25`,borderRadius:6,padding:10}}>
        <div style={{fontSize:11,fontWeight:600,color:t.t0,marginBottom:2}}>{f.issue}</div>
        <div style={{fontSize:10,color:t.acc3,marginBottom:2}}>→ {f.fix}</div>
        <div style={{fontSize:9,color:t.t2,marginBottom:4}}>Impact: {f.impact}</div>
        <div style={{fontSize:9,color:t.t3}}>Affected: {f.layers.join(", ")}</div>
        <button style={{marginTop:6,padding:"4px 12px",borderRadius:4,border:`1px solid ${t.acc3}`,background:t.acc3+"18",color:t.acc3,fontSize:10,fontWeight:600,cursor:"pointer"}}>Apply Fix</button>
      </div>)}
    </div>
  </div>;
}

// ═══════════ HARDWARE REPORT ═══════════
function HWReport({model}){
  const t=useT();const[vendor,setVendor]=useState("all");const[copied,setCopied]=useState(false);
  const totalFlops=model?model.layers.reduce((s,l)=>s+l.flops,0):0;
  const vendors=["all",...[...new Set(HW.map(h=>h.v))].sort()];
  const fil=vendor==="all"?HW:HW.filter(h=>h.v===vendor);
  const scored=fil.map(hw=>{
    const fmtOk=hw.fmts.includes(model?.format||"ONNX");
    const estFPS=Math.min(999,Math.round((hw.tops*1e12)/(totalFlops||1)*(fmtOk?.6:.3)));
    let s=Math.round(hw.sc*(fmtOk?1:.6));
    return{...hw,adjScore:Math.min(100,s),fmtOk,estFPS};
  }).sort((a,b)=>b.adjScore-a.adjScore);

  const report=`EdgeModel Studio — Hardware Report\n${"═".repeat(50)}\nModel: ${model?.name} | ${model?.format} | ${fB(model?.sizeBytes||0)}\nParams: ${fmt(model?.layers.reduce((s,l)=>s+l.params,0)||0)} | FLOPs: ${fmt(totalFlops)}\n${"─".repeat(50)}\n${scored.map((h,i)=>`${(i+1+".").padEnd(4)}${h.n.padEnd(32)} ${h.adjScore}/100  ~${h.estFPS}FPS  ${h.w}W${!h.fmtOk?" ⚠":"  ✓"}`).join("\n")}\n${"─".repeat(50)}\nGenerated ${new Date().toISOString().split("T")[0]}`;

  return<div style={{display:"flex",flexDirection:"column",gap:6,height:"100%"}}>
    <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
      <div style={{fontSize:13,fontWeight:700,color:t.t0}}>Hardware — {scored.length} devices</div>
      <button onClick={()=>{navigator.clipboard.writeText(report);setCopied(true);setTimeout(()=>setCopied(false),2000);}} style={{padding:"3px 10px",borderRadius:4,border:`1px solid ${t.acc}`,background:t.acc+"18",color:t.acc,fontSize:10,fontWeight:600,cursor:"pointer"}}>{copied?"✓ Copied":"Export Report"}</button>
    </div>
    <div style={{display:"flex",gap:3,flexWrap:"wrap"}}>{vendors.map(v=><button key={v} onClick={()=>setVendor(v)} style={{padding:"2px 7px",borderRadius:3,border:`1px solid ${vendor===v?t.acc:t.bdr}`,background:vendor===v?t.acc+"18":"transparent",color:vendor===v?t.acc:t.t2,fontSize:9,cursor:"pointer"}}>{v==="all"?"All":v}</button>)}</div>
    <div style={{flex:1,overflow:"auto",display:"flex",flexDirection:"column",gap:4}}>
      {scored.map(hw=>{
        const bc=hw.adjScore>=80?t.suc:hw.adjScore>=60?t.wrn:t.err;
        return<div key={hw.id} style={{background:t.glass,backdropFilter:"blur(8px)",border:`1px solid ${t.glassBdr}`,borderRadius:6,padding:"8px 10px"}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:3}}>
            <span style={{fontSize:11,fontWeight:600,color:t.t0}}>{hw.n}</span>
            <span style={{fontSize:13,fontWeight:800,color:bc,fontFamily:"'JetBrains Mono',monospace"}}>{hw.adjScore}</span>
          </div>
          <div style={{height:3,background:t.bg3,borderRadius:2,overflow:"hidden",marginBottom:4}}>
            <div style={{width:`${hw.adjScore}%`,height:"100%",background:bc,borderRadius:2,transition:"width .5s"}}/>
          </div>
          <div style={{display:"flex",gap:8,fontSize:9,color:t.t2}}>
            <span>{hw.tops} TOPS</span><span>{hw.ram>0?hw.ram+"GB":"SRAM"}</span><span>{hw.w}W</span><span style={{color:t.acc}}>~{hw.estFPS} FPS</span>
            {!hw.fmtOk&&<span style={{color:t.err}}>needs conversion</span>}
          </div>
        </div>;
      })}
    </div>
  </div>;
}

// ═══════════ CONVERTER ═══════════
function Converter({model}){
  const t=useT();const convs=Object.keys(CONV_SCRIPTS);const[sel,setSel]=useState(convs[0]);
  const[opts,setOpts]=useState({opset:18,sz:640,dyn:false,i8:false,fp16:true,inp:model?.name||"yolo26n.onnx",out:"",prec:"fp16",plat:"rk3588"});
  const script=CONV_SCRIPTS[sel]?.(opts)||"";
  return<div style={{display:"flex",flexDirection:"column",gap:8,height:"100%"}}>
    <div style={{fontSize:13,fontWeight:700,color:t.t0}}>Format Converter</div>
    <div style={{display:"flex",gap:4,flexWrap:"wrap"}}>{convs.map(c=><button key={c} onClick={()=>setSel(c)} style={{padding:"4px 8px",borderRadius:4,border:`1px solid ${sel===c?t.acc:t.bdr}`,background:sel===c?t.acc+"18":t.bg1,color:sel===c?t.acc:t.t2,fontSize:10,fontWeight:600,cursor:"pointer",fontFamily:"'JetBrains Mono',monospace"}}>{c}</button>)}</div>
    <div style={{display:"flex",flexWrap:"wrap",gap:10,padding:8,background:t.glass,backdropFilter:"blur(8px)",borderRadius:6,border:`1px solid ${t.glassBdr}`}}>
      {[["Opset","opset",9,21],["ImgSz","sz",64,1280],].map(([l,k,mn,mx])=><label key={k} style={{display:"flex",alignItems:"center",gap:4,fontSize:10,color:t.t1}}>{l}<input type="number" value={opts[k]} min={mn} max={mx} onChange={e=>setOpts(p=>({...p,[k]:+e.target.value||0}))} style={{width:50,padding:"2px 4px",borderRadius:3,border:`1px solid ${t.bdr}`,background:t.bg,color:t.t0,fontSize:10,fontFamily:"'JetBrains Mono',monospace"}}/></label>)}
      {[["Dynamic","dyn"],["INT8","i8"],["FP16","fp16"]].map(([l,k])=><label key={k} style={{display:"flex",alignItems:"center",gap:4,fontSize:10,color:t.t1,cursor:"pointer"}}><input type="checkbox" checked={opts[k]} onChange={e=>setOpts(p=>({...p,[k]:e.target.checked}))} style={{accentColor:t.acc}}/>{l}</label>)}
    </div>
    <div style={{flex:1,overflow:"auto",position:"relative"}}>
      <button onClick={()=>navigator.clipboard.writeText(script)} style={{position:"absolute",top:6,right:6,padding:"3px 8px",borderRadius:3,border:`1px solid ${t.bdr}`,background:t.bg2,color:t.t2,fontSize:9,cursor:"pointer",zIndex:2}}>Copy</button>
      <pre style={{background:t.bg,border:`1px solid ${t.bdr}`,borderRadius:6,padding:12,fontSize:10.5,color:t.t1,fontFamily:"'JetBrains Mono',monospace",lineHeight:1.6,whiteSpace:"pre-wrap",height:"100%",margin:0,overflow:"auto"}}>{script}</pre>
    </div>
  </div>;
}

// ═══════════ AI CHAT ═══════════
function Chat({model,sel}){
  const t=useT();const[msgs,setMsgs]=useState([]);const[inp,setInp]=useState("");const[ld,setLd]=useState(false);const end=useRef();
  useEffect(()=>{end.current?.scrollIntoView({behavior:"smooth"});},[msgs]);
  const QP=model?["Explain YOLO26 architecture","TensorRT conversion","Deploy on Jetson Orin","Quantization plan","Generate Dockerfile","Benchmark script","INT8 vs FP16","Optimize for Hailo-8"]:[];
  const send=async(text)=>{
    if(!text.trim()||ld)return;const um={role:"user",content:text};const nm=[...msgs,um];setMsgs(nm);setInp("");setLd(true);
    const sys=model?`You are EdgeModel Studio AI — expert Edge AI deployment assistant.\nModel: ${model.name} | ${model.format} | ${model.framework}\nSize: ${fB(model.sizeBytes)} | Input: ${model.inputShape.join("×")} | Params: ${fmt(model.layers.reduce((s,l)=>s+l.params,0))} | FLOPs: ${fmt(model.layers.reduce((s,l)=>s+l.flops,0))}\nKey: YOLO26 = NMS-free, no DFL, end-to-end inference.\n${sel?`Selected: ${sel.name} (${sel.type}), ${sel.shape}`:""}Be concise, technical, actionable. Use code blocks.`:"You are EdgeModel Studio AI.";
    try{const r=await fetch("https://api.anthropic.com/v1/messages",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({model:"claude-sonnet-4-6",max_tokens:1000,system:sys,messages:nm.map(m=>({role:m.role,content:m.content}))})});
      const d=await r.json();setMsgs(p=>[...p,{role:"assistant",content:d.content?.map(b=>b.text||"").join("\n")||"Error."}]);}
    catch(e){setMsgs(p=>[...p,{role:"assistant",content:"Connection error."}]);}setLd(false);
  };
  return<div style={{display:"flex",flexDirection:"column",height:"100%",background:t.bg,borderRadius:6,border:`1px solid ${t.bdr}`,overflow:"hidden"}}>
    <div style={{padding:"6px 10px",borderBottom:`1px solid ${t.bdr}`,fontSize:11,fontWeight:600,color:t.t0,display:"flex",alignItems:"center",gap:5}}>
      <span style={{width:6,height:6,borderRadius:"50%",background:t.suc,display:"inline-block"}}/>AI Assistant
    </div>
    <div style={{flex:1,overflow:"auto",padding:8,display:"flex",flexDirection:"column",gap:5}}>
      {msgs.length===0&&<div style={{color:t.t3,fontSize:10,padding:10,textAlign:"center"}}>{model?"Ask about deployment, optimization, hardware...":"Load a model."}</div>}
      {msgs.map((m,i)=><div key={i} style={{alignSelf:m.role==="user"?"flex-end":"flex-start",maxWidth:"88%",padding:"6px 10px",borderRadius:m.role==="user"?"10px 10px 3px 10px":"10px 10px 10px 3px",background:m.role==="user"?"#1D4ED8":t.bg1,border:m.role==="user"?"none":`1px solid ${t.bdr}`,fontSize:11,color:t.t0,lineHeight:1.5,whiteSpace:"pre-wrap",fontFamily:m.role==="assistant"?"'JetBrains Mono',monospace":"inherit"}}>{m.content}</div>)}
      {ld&&<div style={{alignSelf:"flex-start",padding:"6px 10px",borderRadius:10,background:t.bg1,border:`1px solid ${t.bdr}`,fontSize:10,color:t.t3}}><span style={{animation:"pulse 1.2s infinite",color:t.acc}}>●</span> Analyzing...</div>}
      <div ref={end}/>
    </div>
    {model&&msgs.length===0&&<div style={{padding:"0 8px 4px",display:"flex",flexWrap:"wrap",gap:2}}>{QP.map((p,i)=><button key={i} onClick={()=>send(p)} style={{padding:"2px 7px",borderRadius:8,border:`1px solid ${t.bdr}`,background:"transparent",color:t.t2,fontSize:9,cursor:"pointer"}}>{p}</button>)}</div>}
    <div style={{padding:5,borderTop:`1px solid ${t.bdr}`,display:"flex",gap:4}}>
      <input value={inp} onChange={e=>setInp(e.target.value)} onKeyDown={e=>e.key==="Enter"&&send(inp)} placeholder="Ask anything..." disabled={!model} style={{flex:1,padding:"6px 8px",borderRadius:5,border:`1px solid ${t.bdr}`,background:t.bg1,color:t.t0,fontSize:10,outline:"none"}}/>
      <button onClick={()=>send(inp)} disabled={ld||!model} style={{padding:"6px 12px",borderRadius:5,border:"none",background:ld?t.bg3:t.acc,color:"#fff",fontSize:10,fontWeight:600,cursor:ld?"default":"pointer"}}>Send</button>
    </div>
  </div>;
}

// ═══════════ DEPLOY RECIPES ═══════════
function DeployRecipe({model}){
  const t=useT();const[hw,setHw]=useState("orin_nano");
  const recipes={
    orin_nano:`# ═══ Jetson Orin Nano Deploy Package ═══
# 1. Export to TensorRT
trtexec --onnx=${model?.name||"model.onnx"} --saveEngine=model.engine --fp16 --workspace=4096

# 2. Dockerfile
cat > Dockerfile << 'EOF'
FROM nvcr.io/nvidia/l4t-tensorrt:r8.6.2-runtime
COPY model.engine /app/model.engine
COPY inference.py /app/inference.py
WORKDIR /app
RUN pip install opencv-python-headless numpy
CMD ["python3", "inference.py"]
EOF

# 3. inference.py
cat > inference.py << 'PYEOF'
import tensorrt as trt
import numpy as np, cv2, pycuda.driver as cuda, pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open("model.engine","rb") as f:
    engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(f.read())
context = engine.create_execution_context()
# ... allocate buffers, run inference
print("✓ YOLO26 running on Jetson Orin Nano")
PYEOF

# 4. Benchmark
cat > benchmark.sh << 'EOF'
#!/bin/bash
trtexec --loadEngine=model.engine --warmUp=1000 --iterations=1000 --avgRuns=100
EOF`,
    hailo8:`# ═══ Hailo-8 Deploy Package ═══
pip install hailo_sdk_client

# 1. Convert with Hailo DFC
hailo parser onnx ${model?.name||"model.onnx"} --end-node-names output0
hailo optimize --hw-arch hailo8 --batch-size 1
hailo compile --hw-arch hailo8

# 2. Run with HailoRT
cat > inference.py << 'EOF'
from hailo_platform import HailoRT
target = HailoRT.Device()
hef = HailoRT.HEF("model.hef")
network_group = target.configure(hef)
# ... run inference
print("✓ YOLO26 running on Hailo-8")
EOF`,
    rk3588:`# ═══ RK3588 NPU Deploy ═══
pip install rknn-toolkit2

# 1. Convert
python3 << 'EOF'
from rknn.api import RKNN
rknn = RKNN()
rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform='rk3588')
rknn.load_onnx(model="${model?.name||"model.onnx"}")
rknn.build(do_quantization=True, dataset="calib_list.txt")
rknn.export_rknn("model.rknn")
EOF

# 2. Deploy
adb push model.rknn /data/
adb shell "cd /data && rknn_yolo_demo model.rknn test.jpg"`,
  };
  const targets=[{id:"orin_nano",n:"Jetson Orin Nano"},{id:"hailo8",n:"Hailo-8"},{id:"rk3588",n:"RK3588"}];
  return<div style={{display:"flex",flexDirection:"column",gap:8,height:"100%"}}>
    <div style={{fontSize:13,fontWeight:700,color:t.t0}}>Deployment Recipe Generator</div>
    <div style={{display:"flex",gap:4}}>{targets.map(tg=><button key={tg.id} onClick={()=>setHw(tg.id)} style={{padding:"4px 10px",borderRadius:4,border:`1px solid ${hw===tg.id?t.acc3:t.bdr}`,background:hw===tg.id?t.acc3+"18":"transparent",color:hw===tg.id?t.acc3:t.t2,fontSize:10,cursor:"pointer",fontWeight:500}}>{tg.n}</button>)}</div>
    <div style={{flex:1,overflow:"auto",position:"relative"}}>
      <button onClick={()=>navigator.clipboard.writeText(recipes[hw]||"")} style={{position:"absolute",top:6,right:6,padding:"3px 8px",borderRadius:3,border:`1px solid ${t.bdr}`,background:t.bg2,color:t.t2,fontSize:9,cursor:"pointer",zIndex:2}}>Copy All</button>
      <pre style={{background:t.bg,border:`1px solid ${t.bdr}`,borderRadius:6,padding:12,fontSize:10.5,color:t.suc,fontFamily:"'JetBrains Mono',monospace",lineHeight:1.5,whiteSpace:"pre-wrap",height:"100%",margin:0,overflow:"auto"}}>{recipes[hw]||"Coming soon."}</pre>
    </div>
  </div>;
}

// ═══════════ MAIN APP ═══════════
export default function App(){
  const[theme,setTheme]=useState("dark");const t=TH[theme];
  const[model,setModel]=useState(null);const[sel,setSel]=useState(null);
  const[leftTab,setLeftTab]=useState("graph");const[rightTab,setRightTab]=useState("inspector");
  const[showMath,setShowMath]=useState(true);const[qHeatmap,setQHeatmap]=useState(false);
  const[drag,setDrag]=useState(false);

  const loadDemo=()=>{setModel(DEMO);setSel(null);};
  const handleFile=useCallback((f)=>{if(!f)return;setModel({...DEMO,name:f.name,sizeBytes:f.size,format:f.name.split('.').pop().toUpperCase()});setSel(null);},[]);

  const totalP=model?model.layers.reduce((s,l)=>s+l.params,0):0;
  const totalF=model?model.layers.reduce((s,l)=>s+l.flops,0):0;

  const LTABS=[{id:"graph",label:"Graph",icon:"◇"},{id:"layers",label:"Layers",icon:"☰"}];
  const RTABS=[{id:"inspector",label:"Inspector",icon:"⊞"},{id:"compiler",label:"Compiler",icon:"⚡"},{id:"hardware",label:"Hardware",icon:"⬡"},{id:"convert",label:"Convert",icon:"⇄"},{id:"deploy",label:"Deploy",icon:"🚀"},{id:"chat",label:"AI",icon:"✦"}];

  return<Ctx.Provider value={t}>
    <div style={{height:"100vh",display:"flex",flexDirection:"column",background:t.bg,color:t.t0,fontFamily:"'Inter',-apple-system,sans-serif",overflow:"hidden",transition:"background .3s"}}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700;800&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:4px;height:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:${t.bdr};border-radius:2px}
        @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
        @keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}
        ::selection{background:${t.acc}44}`}</style>

      {/* Header */}
      <header style={{padding:"6px 14px",borderBottom:`1px solid ${t.bdr}`,display:"flex",alignItems:"center",justifyContent:"space-between",flexShrink:0,background:t.glass,backdropFilter:"blur(16px)"}}>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          <div style={{width:28,height:28,borderRadius:7,background:`linear-gradient(135deg,${t.acc},${t.acc2})`,display:"flex",alignItems:"center",justifyContent:"center",fontSize:14,fontWeight:800,color:"#fff",fontFamily:"'Space Grotesk'",boxShadow:`0 0 20px ${t.acc}40`}}>E</div>
          <div>
            <div style={{fontSize:14,fontWeight:800,color:t.t0,fontFamily:"'Space Grotesk'",letterSpacing:-.5}}>EdgeModel Studio</div>
            <div style={{fontSize:8,color:t.t3,letterSpacing:.8,textTransform:"uppercase"}}>AI-Native Model Analyzer · Edge Deployment Intelligence</div>
          </div>
        </div>
        <div style={{display:"flex",alignItems:"center",gap:8}}>
          {model&&<Badge c={t.suc}>{model.format}</Badge>}
          {model&&<span style={{fontSize:10,color:t.t2,fontFamily:"'JetBrains Mono'"}}>{model.name}</span>}
          {model&&<button onClick={()=>{setModel(null);setSel(null);}} style={{padding:"2px 6px",borderRadius:3,border:`1px solid ${t.bdr}`,background:"transparent",color:t.t3,fontSize:9,cursor:"pointer"}}>✕</button>}
          <button onClick={()=>setTheme(p=>p==="dark"?"light":"dark")} style={{width:28,height:28,borderRadius:6,border:`1px solid ${t.bdr}`,background:t.bg2,color:t.t1,fontSize:12,cursor:"pointer",display:"flex",alignItems:"center",justifyContent:"center"}}>{theme==="dark"?"☀":"☾"}</button>
        </div>
      </header>

      {!model?(
        <div style={{flex:1,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",padding:40,gap:28,background:`radial-gradient(ellipse at 50% 40%,${t.acc}08,transparent 70%)`}}>
          <div style={{textAlign:"center"}}>
            <div style={{fontSize:44,fontWeight:800,fontFamily:"'Space Grotesk'",letterSpacing:-2,background:`linear-gradient(135deg,${t.acc},${t.acc2},${t.acc3})`,WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent"}}>EdgeModel Studio</div>
            <div style={{fontSize:13,color:t.t2,marginTop:4}}>AI-native model analyzer · Compiler pre-flight · Edge deployment intelligence</div>
          </div>
          <div onDragOver={e=>{e.preventDefault();setDrag(true);}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);handleFile(e.dataTransfer.files[0]);}}
            style={{width:480,maxWidth:"90vw",padding:40,borderRadius:16,border:`2px dashed ${drag?t.acc:t.bdr}`,background:drag?t.acc+"08":t.glass,backdropFilter:"blur(12px)",textAlign:"center",transition:"all .2s"}}>
            <div style={{fontSize:32,marginBottom:6,opacity:.4}}>⬡</div>
            <div style={{fontSize:15,fontWeight:600,color:t.t0,marginBottom:3}}>Drop your model file</div>
            <div style={{fontSize:11,color:t.t3,marginBottom:16,lineHeight:1.5}}>
              {FORMATS.slice(0,12).join(" · ")}<br/>{FORMATS.slice(12).join(" · ")}
            </div>
            <div style={{display:"flex",gap:8,justifyContent:"center"}}>
              <label style={{padding:"9px 24px",borderRadius:7,background:`linear-gradient(135deg,${t.acc},${t.acc2})`,color:"#fff",fontSize:12,fontWeight:600,cursor:"pointer",boxShadow:`0 0 20px ${t.acc}30`}}>
                Browse Files<input type="file" hidden onChange={e=>handleFile(e.target.files[0])}/>
              </label>
              <button onClick={loadDemo} style={{padding:"9px 24px",borderRadius:7,border:`1px solid ${t.bdr}`,background:"transparent",color:t.t2,fontSize:12,fontWeight:500,cursor:"pointer"}}>Load Demo · YOLO26n</button>
            </div>
          </div>
          <div style={{display:"flex",gap:16,color:t.t3,fontSize:10}}>
            <span>🌐 Web App</span><span>📦 Electron Desktop</span><span>🧩 VS Code Extension</span><span>📱 Mobile Companion</span>
          </div>
        </div>
      ):(
        <>
          {/* Stats */}
          <div style={{padding:"6px 14px",borderBottom:`1px solid ${t.bdr}`,display:"flex",gap:6,overflowX:"auto",flexShrink:0,background:t.glass,backdropFilter:"blur(8px)"}}>
            {[{l:"Params",v:fmt(totalP),c:t.acc},{l:"FLOPs",v:fmt(totalF),c:t.acc2},{l:"Size",v:fB(model.sizeBytes),c:t.suc},{l:"Input",v:model.inputShape.slice(2).join("×"),c:t.wrn},{l:"Layers",v:model.layers.length,c:"#F472B6"},{l:"Opset",v:model.opset,c:t.t2},{l:"NMS-Free",v:"✓",c:t.acc3}].map(s=>
              <div key={s.l} style={{background:t.bg1,border:`1px solid ${t.bdr}`,borderRadius:6,padding:"5px 10px",minWidth:80}}>
                <div style={{fontSize:8,color:t.t3,textTransform:"uppercase",letterSpacing:.7}}>{s.l}</div>
                <div style={{fontSize:15,fontWeight:700,color:s.c,fontFamily:"'JetBrains Mono'",marginTop:1}}>{s.v}</div>
              </div>
            )}
          </div>

          {/* Workspace */}
          <div style={{flex:1,display:"flex",overflow:"hidden"}}>
            {/* Left */}
            <div style={{width:240,borderRight:`1px solid ${t.bdr}`,display:"flex",flexDirection:"column",flexShrink:0,background:t.bg1}}>
              <div style={{padding:"4px 6px",borderBottom:`1px solid ${t.bdr}`,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                <Tabs tabs={LTABS} active={leftTab} onChange={setLeftTab}/>
                <div style={{display:"flex",gap:6,alignItems:"center"}}>
                  <label style={{display:"flex",alignItems:"center",gap:3,fontSize:8,color:t.t3,cursor:"pointer"}} title="Show math"><input type="checkbox" checked={showMath} onChange={e=>setShowMath(e.target.checked)} style={{accentColor:t.acc,width:10,height:10}}/>∑</label>
                  <label style={{display:"flex",alignItems:"center",gap:3,fontSize:8,color:qHeatmap?t.wrn:t.t3,cursor:"pointer"}} title="Quantization heatmap"><input type="checkbox" checked={qHeatmap} onChange={e=>setQHeatmap(e.target.checked)} style={{accentColor:t.wrn,width:10,height:10}}/>Q</label>
                </div>
              </div>
              {leftTab==="graph"?
                <DAGGraph model={model} selected={sel} onSelect={setSel} showMath={showMath} qHeatmap={qHeatmap}/>:
                <div style={{flex:1,overflow:"auto",padding:3}}>
                  {model.layers.map(l=>{const gc=t[GC[l.group]]||t.bdr;const qc=qHeatmap&&l.qSens>0?(l.qSens>0.12?t.err:l.qSens>0.06?t.wrn:t.suc):null;
                    return<div key={l.id} onClick={()=>setSel(l)} style={{padding:"4px 5px",borderRadius:3,cursor:"pointer",marginBottom:1,background:sel?.id===l.id?t.bg3:"transparent",borderLeft:`3px solid ${qc||gc}`}}>
                      <div style={{display:"flex",justifyContent:"space-between"}}><span style={{fontSize:9,fontWeight:600,color:t.t1,fontFamily:"'JetBrains Mono'"}}>{l.type}</span>{l.params>0&&<span style={{fontSize:8,color:t.t3,fontFamily:"'JetBrains Mono'"}}>{fmt(l.params)}</span>}</div>
                      <div style={{fontSize:8,color:t.t3}}>{l.name}</div>
                      <div style={{fontSize:8,color:t.acc,fontFamily:"'JetBrains Mono'"}}>{l.shape}</div>
                    </div>;
                  })}
                </div>
              }
            </div>

            {/* Right */}
            <div style={{flex:1,display:"flex",flexDirection:"column",overflow:"hidden"}}>
              <div style={{padding:"4px 10px",borderBottom:`1px solid ${t.bdr}`,background:t.bg1}}><Tabs tabs={RTABS} active={rightTab} onChange={setRightTab}/></div>
              <div style={{flex:1,overflow:"auto",padding:10}}>
                {rightTab==="inspector"&&<Inspector layer={sel} model={model}/>}
                {rightTab==="compiler"&&<CompilerChecker model={model}/>}
                {rightTab==="hardware"&&<HWReport model={model}/>}
                {rightTab==="convert"&&<Converter model={model}/>}
                {rightTab==="deploy"&&<DeployRecipe model={model}/>}
                {rightTab==="chat"&&<div style={{height:"100%"}}><Chat model={model} sel={sel}/></div>}
              </div>
            </div>
          </div>

          {/* Footer */}
          <div style={{padding:"3px 14px",borderTop:`1px solid ${t.bdr}`,display:"flex",justifyContent:"space-between",fontSize:8,color:t.t3,background:t.glass,backdropFilter:"blur(8px)",flexShrink:0}}>
            <span>EdgeModel Studio v3.0 · {model.format} · opset {model.opset} · {model.framework} · {model.producer}</span>
            <span>Phase 1: Graph + Inspector + Compiler Checker + Auto-Fix + Deploy Recipes · 🌐 Web · 📦 Desktop · 🧩 VSCode</span>
          </div>
        </>
      )}
    </div>
  </Ctx.Provider>;
}
