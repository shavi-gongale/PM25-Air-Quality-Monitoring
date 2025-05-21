package com.pm25app.android;

import android.content.Context;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Bundle;
import android.view.View;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    private WebView webView;
    private ProgressBar progressBar;
    private RelativeLayout offlineLayout;
    private Button retryButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Initialize views
        webView = findViewById(R.id.webView);
        progressBar = findViewById(R.id.progressBar);
        offlineLayout = findViewById(R.id.offlineLayout);
        retryButton = findViewById(R.id.retryButton);

        // Configure WebView
        setupWebView();

        // Set retry button click listener
        retryButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                checkNetworkAndLoadUrl();
            }
        });

        // Check network and load URL
        checkNetworkAndLoadUrl();
    }

    private void setupWebView() {
        WebSettings webSettings = webView.getSettings();
        webSettings.setJavaScriptEnabled(true);
        webSettings.setDomStorageEnabled(true);
        webSettings.setLoadWithOverviewMode(true);
        webSettings.setUseWideViewPort(true);
        webSettings.setSupportZoom(true);
        webSettings.setBuiltInZoomControls(true);
        webSettings.setDisplayZoomControls(false);
        
        // Improve WebView performance
        webSettings.setRenderPriority(WebSettings.RenderPriority.HIGH);
        webSettings.setCacheMode(WebSettings.LOAD_DEFAULT);
        webSettings.setAppCacheEnabled(true);
        
        webView.setWebViewClient(new WebViewClient() {
            @Override
            public void onPageFinished(WebView view, String url) {
                super.onPageFinished(view, url);
                // Hide progress bar when page is loaded
                progressBar.setVisibility(View.GONE);
                
                // Apply CSS for better mobile UI
                applyMobileOptimizationCSS(view);
            }

            @Override
            public void onReceivedError(WebView view, WebResourceRequest request, WebResourceError error) {
                super.onReceivedError(view, request, error);
                if (isNetworkConnected()) {
                    // If there's an error loading the page but we have network,
                    // it might be a server issue
                    showOfflineMode();
                }
            }
        });
    }

    private void applyMobileOptimizationCSS(WebView view) {
        String css = "javascript:(function() {" +
                    "var style = document.createElement('style');" +
                    "style.innerHTML = '" +
                    "body { font-size: 16px !important; padding: 10px !important; }" +
                    ".stButton button { width: 100% !important; }" +
                    ".stSelectbox div[data-baseweb=\"select\"] { width: 100% !important; }" +
                    ".stTextInput input { width: 100% !important; }" +
                    "div.block-container { padding: 1rem 0.5rem !important; max-width: 100% !important; }" +
                    "';" +
                    "document.head.appendChild(style);" +
                    "})()";
        view.loadUrl(css);
    }

    private void checkNetworkAndLoadUrl() {
        if (isNetworkConnected()) {
            showWebView();
            // Load the Streamlit app URL
            // In production, this would be your deployed Streamlit app URL
            webView.loadUrl("http://10.0.2.2:5000"); // This maps to host's localhost in Android emulator
            // For real device testing, use your actual server IP or domain:
            // webView.loadUrl("https://your-streamlit-deployment-url.com");
        } else {
            showOfflineMode();
        }
    }

    private boolean isNetworkConnected() {
        ConnectivityManager connectivityManager = (ConnectivityManager) getSystemService(Context.CONNECTIVITY_SERVICE);
        if (connectivityManager != null) {
            NetworkInfo activeNetworkInfo = connectivityManager.getActiveNetworkInfo();
            return activeNetworkInfo != null && activeNetworkInfo.isConnected();
        }
        return false;
    }

    private void showWebView() {
        webView.setVisibility(View.VISIBLE);
        progressBar.setVisibility(View.VISIBLE);
        offlineLayout.setVisibility(View.GONE);
    }

    private void showOfflineMode() {
        webView.setVisibility(View.GONE);
        progressBar.setVisibility(View.GONE);
        offlineLayout.setVisibility(View.VISIBLE);
    }

    @Override
    public void onBackPressed() {
        if (webView.canGoBack()) {
            webView.goBack();
        } else {
            super.onBackPressed();
        }
    }
}