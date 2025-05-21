# PM2.5 Air Quality Android App

This Android application is a mobile wrapper for the PM2.5 Air Quality Prediction web application. It provides access to all the features of the web application in a native Android app format.

## Features

- Real-time PM2.5 air quality monitoring
- Machine learning-based predictions
- Historical data analysis
- Satellite imagery integration
- User-friendly interface with dark/light mode
- Works offline (displays cached data when no internet connection)

## Building the App

### Prerequisites

- Android Studio 4.0 or higher
- JDK 8 or higher
- Android SDK with API level 21 or higher

### Steps to Build

1. Open the project in Android Studio
2. Let Gradle sync the project
3. Connect an Android device or use an emulator
4. Click "Run" to build and install the app

## Deployment

To create a release APK for distribution:

1. In Android Studio, go to Build → Generate Signed Bundle/APK
2. Select "APK"
3. Create or use an existing keystore
4. Select "release" build variant
5. Click "Finish" to generate the APK
6. The APK will be located in `android_app/app/release/app-release.apk`

## How to Use

1. Install the APK on your Android device
2. Launch the app
3. Allow internet permissions when prompted
4. The app will load the PM2.5 Air Quality Prediction web interface
5. Use all features available in the web version

## Technical Details

- The app uses WebView to display the web application
- Additional native features include:
  - Splash screen
  - Offline detection
  - Screen orientation handling
  - Custom app icon
  
## Customization

To change the URL of the web application:

1. Open `MainActivity.java`
2. Locate the `checkNetworkAndLoadUrl()` method
3. Change the URL in the `webView.loadUrl()` call

```java
// Replace with your actual Streamlit deployment URL
webView.loadUrl("https://your-streamlit-app-url.com");
```