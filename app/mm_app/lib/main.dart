import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Camera App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: MyHomePage(title: 'Camera App'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key key, this.title}) : super(key: key);
  final String title;
  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
          child:
          (_cameraInitialized)
              ? AspectRatio(aspectRatio: _camera.value.aspectRatio,
            child: CameraPreview(_camera),)
              : CircularProgressIndicator()
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: (){},
        tooltip: 'Increment',
        child: Icon(Icons.camera_alt),
      ), // This trailing comma makes auto-formatting nicer for build methods.
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }
  CameraController _camera;
  bool _cameraInitialized = false;
  CameraImage _savedImage;
  void _initializeCamera() async {
    // Get list of cameras of the device
    List<CameraDescription> cameras = await availableCameras();
// Create the CameraController
    _camera = new CameraController(
        cameras[0], ResolutionPreset.veryHigh
    );
// Initialize the CameraController
    _camera.initialize().then((_) async{
      // Start ImageStream
      await _camera.startImageStream((CameraImage image) =>
          _processCameraImage(image));
      setState(() {
        _cameraInitialized = true;
      });
    });
  }
  void _processCameraImage(CameraImage image) async {
    setState(() {
      _savedImage = image;
    });
  }
  @override
  void initState(){
    super.initState();
    _initializeCamera();
  }
}

