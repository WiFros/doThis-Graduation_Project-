package fragment

import android.content.Intent
import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import androidx.databinding.DataBindingUtil.setContentView
import com.example.healthcare_exercise.MainActivity
import com.example.healthcare_exercise.R
import com.example.healthcare_exercise.databinding.FragmentMyPageBinding

class MyPageFragment : Fragment() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        val btn_signOut = getView()?.findViewById<Button>(R.id.btn_signOut)
        btn_signOut?.setOnClickListener(View.OnClickListener {
            (activity as MainActivity).signOut()
            val intent = Intent(context,MainActivity::class.java)
            startActivity(intent)
        })
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        // Inflate the layout for this fragment

        return inflater.inflate(R.layout.fragment_my_page, container, false)
    }
}