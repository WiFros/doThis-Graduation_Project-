package com.example.healthcare_exercise

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import com.example.healthcare_exercise.databinding.ActivityExerciseBinding
import fragment.ExerciseFragment

//import fragment.ptFragment

class ExerciseActivity : AppCompatActivity() {

    private var mBinding: ActivityExerciseBinding? = null
    private val binding get() = mBinding!!

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        mBinding = ActivityExerciseBinding.inflate(layoutInflater)
        setContentView(binding.root)


    }

    fun ptList(view: View) {

    }
}