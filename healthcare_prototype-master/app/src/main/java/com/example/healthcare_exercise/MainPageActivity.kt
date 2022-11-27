package com.example.healthcare_exercise

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import com.example.healthcare_exercise.databinding.ActivityMainPageBinding
import com.google.android.material.bottomnavigation.BottomNavigationView
import fragment.BalanceFragment
import fragment.CommunityFragment
import fragment.ExerciseFragment
import fragment.MyPageFragment

class MainPageActivity : AppCompatActivity() {

    private var mBinding: ActivityMainPageBinding? = null
    private val binding get() = mBinding!!

    lateinit var email:String
    lateinit var name:String

    private val PERMISSIONS_REQUEST_RESULT = 1

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        mBinding = ActivityMainPageBinding.inflate(layoutInflater)
        setContentView(binding.root)

        //읽어온 사용자 정보 받아오기
        email = intent.getStringExtra("email").toString()
        name = intent.getStringExtra("name").toString()

        //default fragment 설정
        setFragment(ExerciseFragment())

        //bottom navigation fragment 변경
        var bottomNav = findViewById<BottomNavigationView>(R.id.bottomNav)
        bottomNav.run { setOnNavigationItemReselectedListener {
            when(it.itemId){
                R.id.bn_exercise ->{
                    Toast.makeText(context, "E clicked", Toast.LENGTH_SHORT).show()
                    setFragment(ExerciseFragment())
                }
                R.id.bn_balance ->{
                    Toast.makeText(context, "B clicked", Toast.LENGTH_SHORT).show()
                    setFragment(BalanceFragment())
                }
                R.id.bn_community ->{
                    Toast.makeText(context, "C clicked", Toast.LENGTH_SHORT).show()
                    setFragment(CommunityFragment())
                }
                R.id.bn_mypage ->{
                    Toast.makeText(context, "M clicked", Toast.LENGTH_SHORT).show()
                    setFragment(MyPageFragment())
                }
            }
            true
        } }

        //접근 권환
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)!= PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE), PERMISSIONS_REQUEST_RESULT)
        }
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)!= PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE), PERMISSIONS_REQUEST_RESULT)
        }
        if(ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED){
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), PERMISSIONS_REQUEST_RESULT)
        }

//        upload 접근 권한 설정
//        ActivityCompat.requestPermissions(this,
//            arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE),1)
//        ActivityCompat.requestPermissions(this,
//            arrayOf(Manifest.permission.CAMERA),1)
//        ActivityCompat.requestPermissions(this,
//            arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE),1)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSIONS_REQUEST_RESULT){
            if(grantResults[0] == PackageManager.PERMISSION_GRANTED)
                Toast.makeText(this, "권한 요청이 되었습니다",Toast.LENGTH_LONG).show()
            else {
                Toast.makeText(this, "권한 요청을 해주세요", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }

    fun setDataAtFragment(fragment:Fragment, uri:String){
        val bundle = Bundle()
        bundle.putString("name",name)
        bundle.putString("email",email)
        bundle.putString("uri",uri)

        fragment.arguments=bundle
        setFragment(fragment)
    }

    fun setFragment(fragment: Fragment){
        supportFragmentManager.beginTransaction().replace(R.id.mainFrame, fragment).commit()
    }

    override fun onBackPressed() {
        //super.onBackPressed()
    }
}