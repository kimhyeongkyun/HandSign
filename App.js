import React, { Component } from 'react';
import { Dimensions, Modal, SafeAreaView, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { Camera } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';
import Svg, { Circle, Line } from 'react-native-svg';
import * as posenet from '@tensorflow-models/posenet';
import Axios from 'axios';

// 전체 화면 변수
const fullWidth = Dimensions.get("window").width;
const fullheight = Dimensions.get("window").height;

const inputTensorWidth = 152;
const inputTensorHeight = 200;

//기본 카메라를 이용해 tensorflow에서 제공해주는 함수를 이용해 tensorcamera를 만듬
const TensorCamera = cameraWithTensors(Camera);
class App extends Component {
  constructor(props) {
    super(props);
    //state
    this.state = {
      isTfReady: false,
      pose: null,
      rafID: -1,
      isStart: false,
    };
    // fncHandleCameraStream함수 bind
    this.fncHandleCameraStream = this.fncHandleCameraStream.bind(this);
  }
  // Component가 mount 된 직후
  async componentDidMount() {
    //mount 될때 tensorflow를 사용하기위해 ready한다.
    await tf.ready();
    // state isTfReady 를 변경해준다.
    this.setState({ isTfReady: true });
  }
  //component가 제거되기 직전에 불려짐
  componentWillUnmount() {
    // rafID가 -1이 아닐 때 cancelAnimationFrame를 사용해  rafID를 제거한다.
    if (this.state.rafID !== -1) {
      cancelAnimationFrame(this.state.rafID);
    }
  }

  // poseData를 백엔드로 post 해준다. 
  async fncPostPoseData() {
    const result = await Axios.post("http://172.30.1.49:3000/poseData", { data: this.state.pose, isStart: this.state.isStart }).then(
      function (response) {
        // console.log(response);
      }
    );
  }

  // camera가 실행 될때 호출 되는 함수
  //https://js.tensorflow.org/api_react_native/0.2.1/
  //https://github.com/tensorflow/tfjs/blob/master/tfjs-react-native/integration_rn59/components/webcam/realtime_demo.tsx
  async fncHandleCameraStream(images, updatePreview, gl) {
    const loop = async () => {
      updatePreview();
      // 필요에 따라 카메라 이미지를 나타내는 tenstor를 생성
      const nextImageTensor = images.next().value
      if (nextImageTensor) {
        const flipHorizontal = Platform.OS === 'ios' ? false : true;
        //posenet 모델을 사용
        const model = await posenet.load({ architecture: 'MobileNetV1', outputStride: 16, inputResolution: { width: inputTensorWidth, height: inputTensorHeight }, multiplier: 0.75, quantBytes: 2 });
        // pose 변수에 posenet model estimateSinglePose함수를 사용해 pose data를 넣는다. estimateSinglePose함수는 사람이 한명일때 사용한다.
        // 파라미터로는 nexImageTensor와 ios일 경우에는 flipHorizontal을 false로 설정한다.
        const pose = await model.estimateSinglePose(nextImageTensor, { flipHorizontal });
        // 받아온 pose를 state에 저장한다.
        this.setState({ pose: pose });
        console.log(this.state.pose)
        // tf.dispose함수를 이용해 tensor를 제거해준다.
        tf.dispose([nextImageTensor]);
      }
      gl.endFrameEXP();
      // rafID에 requestAnimationFrame id를 지정해 저장해준다.
      this.setState({ rafID: requestAnimationFrame(loop) });
      // pose data post 함수 호출
      this.fncPostPoseData();
    }
    // loop
    loop();

  }

  fncHandleButton(isStart) {
    if (isStart === true) {
      this.setState({ isStart: false })
    } else {
      this.setState({ isStart: true })
    }
  }

  async fncAnolysis() {
    // rest api 호출
    const result = await Axios.post("http://172.30.1.45:3000/analysis", { anolysis: true }).then(
      function (response) {
        console.log(response);
      }
    );
  }

  //눈, 코, 입, 귀, 팔 등 점과 선이 render되는 코드 
  renderPose() {
    const MIN_KEYPOINT_SCORE = 0.2;
    const { pose } = this.state;
    // state의 pose가 있을 경우에만 Render
    if (pose != null) {
      // keypoint(눈, 코, 입 등) 점으로 화면에 render됨
      const keypoints = pose.keypoints.filter(k => k.score > MIN_KEYPOINT_SCORE).map((k, i) => {
        return (<Circle key={`skeletonkp_${i}`} cx={k.position.x} cy={k.position.y} r='2' strokeWidth='0' fill='blue' />)
      });
      const adjacentKeypoints = posenet.getAdjacentKeyPoints(pose.keypoints, MIN_KEYPOINT_SCORE);
      //점과 점사이를 이어주는 선render
      const skeleton = adjacentKeypoints.map(([from, to], i) => {
        return (<Line key={`skeletonls_${i}`} x1={from.position.x} y1={from.position.y} x2={to.position.x} y2={to.position.y} stroke='magenta' strokeWidth='1' />)
      });
      return <Svg height='100%' width='100%' viewBox={`0 0 ${inputTensorWidth} ${inputTensorHeight}`}>{skeleton}{keypoints}</Svg>;
    } else {
      return null;
    }
  }

  //기본으로 render되는 코드
  render() {
    let textureDims;
    if (Platform.OS === 'ios') {
      textureDims = { height: 1920, width: 1080, };
    } else {
      textureDims = { height: 1200, width: 1600, };
    }
    let { isStart } = this.state;
    return (
      // 전체 화면
      <View style={styles.canvas}>
        {/* 카메라 */}
        <TensorCamera style={styles.cameraStyle} type={Camera.Constants.Type.front} cameraTextureHeight={textureDims.height} cameraTextureWidth={textureDims.width} resizeHeight={200}
          resizeWidth={152} resizeDepth={3} onReady={this.fncHandleCameraStream} autorender={true} />
        {/* 얼굴위 선과 점  */}
        <View style={styles.modelStyle}>
          {this.renderPose()}
        </View>
        <View style={styles.buttonLayerStyle}>
          {/* 시작 종료 버튼 */}
          <TouchableOpacity onPress={() => { this.fncHandleButton(isStart) }} style={styles.buttonStyle}>
            {isStart === false ? (<Text style={styles.buttonTextStyle}>시작</Text>) : (<Text style={styles.buttonTextStyle}>종료</Text>)}
          </TouchableOpacity>
          {/* 분석 버튼 */}
          <TouchableOpacity disabled={this.state.isStart} onPress={() => { this.fncAnolysis() }} style={styles.buttonStyle}>
            <Text style={styles.buttonTextStyle}>분석</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }
}
//style 정리해둔 코드 
const styles = StyleSheet.create({
  canvas: { flex: 1, alignItems: "center", justifyContent: "center" },
  cameraStyle: { position: 'absolute', width: fullWidth - 20, height: fullheight - 250, zIndex: 0, borderWidth: 1 },
  modelStyle: { position: 'absolute', width: fullWidth - 20, height: fullheight - 250, zIndex: 20, borderWidth: 1 },
  buttonLayerStyle: { position: "absolute", bottom: 60, flexDirection: "row" },
  buttonStyle: { alignSelf: "flex-end", backgroundColor: "black", width: 100, height: 50, alignItems: "center", justifyContent: "center", marginHorizontal: 10 },
  buttonTextStyle: { fontSize: 20, color: "white", fontWeight: '600' }
});
export default App;