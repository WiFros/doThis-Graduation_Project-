package com.example.healthcare_exercise

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Toast
import com.example.healthcare_exercise.databinding.ActivitySignInBinding

class SignInActivity : AppCompatActivity() {
    private var mBinding:ActivitySignInBinding? = null
    private val binding get() = mBinding!!
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        mBinding = ActivitySignInBinding.inflate(layoutInflater)
        setContentView(binding.root)
    }

    fun submit(view: View) {
        //항목들을 확인 후 storage와 연동.
        //이상 없으면 회원가입 완료
        //로그인 화면으로 돌아감
        Toast.makeText(this, "Sign In Success!",Toast.LENGTH_SHORT).show()
        val intent = Intent(this,MainActivity::class.java)
        startActivity(intent)
    }
}