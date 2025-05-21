package com.pm25app.android;

import android.content.Context;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.net.ConnectivityManager;
import android.net.NetworkInfo;
import android.os.Bundle;
import android.text.InputType;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.webkit.CookieManager;
import android.webkit.WebResourceError;
import android.webkit.WebResourceRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import android.webkit.WebViewClient;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.Toast;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {

    private WebView webView;
    private ProgressBar progressBar;
    private RelativeLayout offlineLayout;
    private Button retryButton;
    
    private static final String PREFS_NAME = "PM25AppPrefs";
    private static final String API_KEY_PREF = "openweather_api_key";
    private SharedPreferences sharedPreferences;
    private String openWeatherApiKey;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        // Set up ActionBar
        setSupportActionBar(findViewById(R.id.toolbar));

        // Initialize shared preferences
        sharedPreferences = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        openWeatherApiKey = sharedPreferences.getString(API_KEY_PREF, "");

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

        // Check if API key is missing and prompt user
        if (openWeatherApiKey.isEmpty()) {
            showApiKeyDialog();
        } else {
            // Check network and load URL
            checkNetworkAndLoadUrl();
        }
    }
    
    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main_menu, menu);
        return true;
    }
    
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        int id = item.getItemId();
        
        if (id == R.id.action_api_key) {
            showApiKeyDialog();
            return true;
        }
        
        return super.onOptionsItemSelected(item);
    }
    
    private void showApiKeyDialog() {
        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("OpenWeatherMap API Key");
        builder.setMessage("Enter your OpenWeatherMap API key to fetch real weather data");

        // Set up the input
        final EditText input = new EditText(this);
        input.setInputType(InputType.TYPE_CLASS_TEXT);
        if (!openWeatherApiKey.isEmpty()) {
            input.setText(openWeatherApiKey);
        }
        builder.setView(input);

        // Set up the buttons
        builder.setPositiveButton("Save", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                openWeatherApiKey = input.getText().toString().trim();
                if (!openWeatherApiKey.isEmpty()) {
                    // Save API key to SharedPreferences
                    SharedPreferences.Editor editor = sharedPreferences.edit();
                    editor.putString(API_KEY_PREF, openWeatherApiKey);
                    editor.apply();
                    
                    Toast.makeText(MainActivity.this, "API key saved", Toast.LENGTH_SHORT).show();
                    
                    // Clear cookies and reload page to use the new API key
                    CookieManager.getInstance().removeAllCookies(null);
                    checkNetworkAndLoadUrl();
                } else {
                    Toast.makeText(MainActivity.this, "API key cannot be empty", Toast.LENGTH_SHORT).show();
                }
            }
        });
        
        builder.setNegativeButton("Cancel", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                dialog.cancel();
                if (openWeatherApiKey.isEmpty()) {
                    // If still no API key, load the page anyway
                    checkNetworkAndLoadUrl();
                }
            }
        });

        builder.show();
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
            
            // Set cookies to pass API key to Streamlit app
            if (!openWeatherApiKey.isEmpty()) {
                CookieManager cookieManager = CookieManager.getInstance();
                cookieManager.setAcceptCookie(true);
                
                // Set the cookie with the API key (domain must match your URL)
                String cookieString = "openweather_api_key=" + openWeatherApiKey;
                
                // For emulator
                cookieManager.setCookie("10.0.2.2", cookieString);
                
                // For production URL
                // cookieManager.setCookie("your-streamlit-deployment-url.com", cookieString);
                
                cookieManager.flush();
            }
            
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