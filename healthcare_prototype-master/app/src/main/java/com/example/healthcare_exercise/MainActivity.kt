package com.example.healthcare_exercise

import android.content.Context
import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import com.example.healthcare_exercise.databinding.ActivityMainBinding
import com.google.android.gms.auth.api.signin.GoogleSignIn
import com.google.android.gms.auth.api.signin.GoogleSignInAccount
import com.google.android.gms.auth.api.signin.GoogleSignInClient
import com.google.android.gms.auth.api.signin.GoogleSignInOptions
import com.google.android.gms.common.api.ApiException
import com.google.android.gms.tasks.Task
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.FirebaseUser
import com.google.firebase.auth.GoogleAuthProvider
import com.google.firebase.auth.ktx.auth
import com.google.firebase.ktx.Firebase
import com.google.firebase.storage.FirebaseStorage

class MainActivity : AppCompatActivity() {

    private var mBinding: ActivityMainBinding? = null
    private val binding get() = mBinding!!

    private lateinit var auth: FirebaseAuth
    private lateinit var googleSignInClient: GoogleSignInClient

    lateinit var mContext: Context
    lateinit var email: String
    lateinit var displayName: String

    //==================================================================================================
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        mBinding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        mContext = this;

        // [START config_signin]
        // Configure Google Sign In
        val gso = GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
            .requestIdToken(getString(R.string.default_web_client_id))
            .requestEmail()
            .build()


        googleSignInClient = GoogleSignIn.getClient(this, gso)
        println("good")
        // [END config_signin]


        // [START initialize_auth]
        // Initialize Firebase Auth
        auth = Firebase.auth
        // [END initialize_auth]

        mBinding!!.btnGoogleSignIn.setOnClickListener(View.OnClickListener {
            val signInIntent = googleSignInClient.signInIntent
            startActivityForResult(signInIntent, RC_SIGN_IN)
            println("clicked")
        })
    }

//    override fun onStart() {
//        super.onStart()
//        // Check if user is signed in (non-null) and update UI accordingly.
//        //자동로그인
//        val currentUser = auth.currentUser
//        updateUI(currentUser)
//    }
    // [END on_start_check_user]

    // [START onactivityresult]
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        // Result returned from launching the Intent from GoogleSignInApi.getSignInIntent(...);
        if (requestCode == RC_SIGN_IN) {
            val task = GoogleSignIn.getSignedInAccountFromIntent(data)
            try {
                // Google Sign In was successful, authenticate with Firebase
                val account = task.getResult(ApiException::class.java)!!
                Log.d(TAG, "firebaseAuthWithGoogle:" + account.id)
                val task: Task<GoogleSignInAccount> = GoogleSignIn.getSignedInAccountFromIntent(data)
                handleSignInResult(task)
                firebaseAuthWithGoogle(account.idToken!!)
            } catch (e: ApiException) {
                // Google Sign In failed, update UI appropriately
                Log.w(TAG, "Google sign in failed", e)
            }
        }
    }
    // [END onactivityresult]

    // [START auth_with_google]
    private fun firebaseAuthWithGoogle(idToken: String) {
        val credential = GoogleAuthProvider.getCredential(idToken, null)
        auth.signInWithCredential(credential)
            .addOnCompleteListener(this) { task ->
                if (task.isSuccessful) {
                    // Sign in success, update UI with the signed-in user's information
                    Log.d(TAG, "signInWithCredential:success")
                    val user = auth.currentUser
                    updateUI(user)
                } else {
                    // If sign in fails, display a message to the user.
                    Log.w(TAG, "signInWithCredential:failure", task.exception)
                    updateUI(null)
                }
            }
    }
    // [END auth_with_google]

    private fun updateUI(user: FirebaseUser?) {
        val intent = Intent(this, MainPageActivity::class.java)
        intent.putExtra("email",email)
        intent.putExtra("name",displayName)
        startActivity(intent)
    }

    fun signOut(){
        Firebase.auth.signOut()
        googleSignInClient!!.signOut().addOnCompleteListener{
            MainPageActivity().finish()
            MainActivity().finish()
        }
        Toast.makeText(this, "sign Out!",Toast.LENGTH_LONG).show()
    }

    private fun handleSignInResult(completedTask: Task<GoogleSignInAccount>) {
        try {
            val account = completedTask.getResult(ApiException::class.java)
            email = account?.email.toString()
            displayName = account?.displayName.toString()

            Log.d("account", email)
            Log.d("account", displayName)
        } catch (e: ApiException) {
            // The ApiException status code indicates the detailed failure reason.
            // Please refer to the GoogleSignInStatusCodes class reference for more information.
            Log.w("failed", "signInResult:failed code=" + e.statusCode)
        }
    }
    companion object {
        private const val TAG = "GoogleActivity"
        private const val RC_SIGN_IN = 9001
    }
}



//==================================================================================================
//    override fun onCreate(savedInstanceState: Bundle?) {
//        super.onCreate(savedInstanceState)
//
//        mBinding = ActivityMainBinding.inflate(layoutInflater)
//        setContentView(binding.root)
//
//        mContext = this;
//
//        // [START config_signin]
//        // Configure Google Sign In
//        val gso = GoogleSignInOptions.Builder(GoogleSignInOptions.DEFAULT_SIGN_IN)
//            .requestIdToken(getString(R.string.default_web_client_id))
//            .requestEmail()
//            .build()
//
//
//        googleSignInClient = GoogleSignIn.getClient(this, gso)
//        println("good")
//        // [END config_signin]
//
//
//        // [START initialize_auth]
//        // Initialize Firebase Auth
//        auth = Firebase.auth
//        // [END initialize_auth]
//
//        mBinding!!.btnGoogleSignIn.setOnClickListener(View.OnClickListener {
//            val signInIntent = googleSignInClient.signInIntent
//            startActivityForResult(signInIntent, RC_SIGN_IN)
//            println("clicked")
//        })
//    }
//
//    override fun onStart() {
//        super.onStart()
//        // Check if user is signed in (non-null) and update UI accordingly.
//        val currentUser = auth.currentUser
//        updateUI(currentUser)
//    }
//    // [END on_start_check_user]
//
//    // [START onactivityresult]
//    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
//        super.onActivityResult(requestCode, resultCode, data)
//
//        // Result returned from launching the Intent from GoogleSignInApi.getSignInIntent(...);
//        if (requestCode == RC_SIGN_IN) {
//            val task = GoogleSignIn.getSignedInAccountFromIntent(data)
//            try {
//                // Google Sign In was successful, authenticate with Firebase
//                val account = task.getResult(ApiException::class.java)!!
//                Log.d(TAG, "firebaseAuthWithGoogle:" + account.id)
//                firebaseAuthWithGoogle(account.idToken!!)
//            } catch (e: ApiException) {
//                // Google Sign In failed, update UI appropriately
//                Log.w(TAG, "Google sign in failed", e)
//            }
//        }
//    }
//    // [END onactivityresult]
//
//    // [START auth_with_google]
//    private fun firebaseAuthWithGoogle(idToken: String) {
//        val credential = GoogleAuthProvider.getCredential(idToken, null)
//        auth.signInWithCredential(credential)
//            .addOnCompleteListener(this) { task ->
//                if (task.isSuccessful) {
//                    // Sign in success, update UI with the signed-in user's information
//                    Log.d(TAG, "signInWithCredential:success")
//                    val user = auth.currentUser
//                    updateUI(user)
//                } else {
//                    // If sign in fails, display a message to the user.
//                    Log.w(TAG, "signInWithCredential:failure", task.exception)
//                    getUserProfile()
//                    updateUI(null)
//                }
//            }
//    }
//    // [END auth_with_google]
//
//    private fun updateUI(user: FirebaseUser?) {
//        val intent = Intent(this, MainPageActivity::class.java)
//        startActivity(intent)
//    }
//
//    fun signOut(){
//        Firebase.auth.signOut()
//        googleSignInClient!!.signOut().addOnCompleteListener{
//            MainPageActivity().finish()
//            MainActivity().finish()
//        }
//        Toast.makeText(this, "sign Out!",Toast.LENGTH_LONG).show()
//    }
//
//    private fun getUserProfile(){
//        val user = Firebase.auth.currentUser
//        Toast.makeText(this,"GET START",Toast.LENGTH_SHORT).show()
////        user?.let {
//        Toast.makeText(this,"GET INININ",Toast.LENGTH_SHORT).show()
//        val name = user?.displayName
//        val email = user?.email
//        val photoUrl = user?.photoUrl
//
////            val emailVerified = user.isEmailVerified
//
////            val uid = user.uid
//        Toast.makeText(this,name+"----"+email,Toast.LENGTH_SHORT).show()
//
//        Log.i("aa", "=============================%%%%%%#$%#%#$%#$%#$%#$%#$%#$%#$%")
//        Log.i("aa", "getUserProfile: "+email+" "+name)
////        }
//    }
//
//    companion object {
//        private const val TAG = "GoogleActivity"
//        private const val RC_SIGN_IN = 9001
//    }
//}


//==================================================================================================




//    //입력된 id, pw 읽어와서 파이어베이스와 비교. 로그인으로 넘어감.
//    fun login(view: View) {
//        val id = binding.etId.toString()
//        val pw = binding.etPw.toString()
//
//        val intent = Intent(this, MainPageActivity::class.java)
//        startActivity(intent)
//    }
//
//    fun signIn(view: View) {
//        val intent = Intent(this, SignInActivity::class.java)
//        startActivity(intent)
//    }