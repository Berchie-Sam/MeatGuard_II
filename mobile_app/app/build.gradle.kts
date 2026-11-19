plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.meatguard.app"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.meatguard.app"
        minSdk = 24
        targetSdk = 36
        versionCode = 3
        versionName = "1.2-research"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildFeatures {
        viewBinding = true
        buildConfig = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    androidResources {
        // The model is memory-mapped, so it must be stored uncompressed in the APK.
        noCompress.add("tflite")
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.activity)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.lifecycle.runtime)
    implementation(libs.androidx.exifinterface)
    implementation(libs.gson)
    implementation(libs.tflite)
    implementation(libs.tflite.gpu)
    // GpuDelegateFactory$Options lives here; without it GpuDelegate() fails with NoClassDefFoundError.
    implementation(libs.tflite.gpu.api)
    implementation(libs.image.cropper)

    testImplementation(libs.junit)
}
