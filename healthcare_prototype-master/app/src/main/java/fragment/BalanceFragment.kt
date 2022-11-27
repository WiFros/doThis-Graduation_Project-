package fragment

import android.content.Intent
import android.graphics.Color
import android.graphics.Paint
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.MediaController
import android.widget.Toast
import androidx.core.content.FileProvider
import com.example.healthcare_exercise.MainPageActivity
import com.example.healthcare_exercise.R
import com.google.firebase.storage.FirebaseStorage
import kotlinx.android.synthetic.main.fragment_balance.*
import kotlinx.android.synthetic.main.fragment_balance.view.*
import kotlinx.android.synthetic.main.fragment_exercise_upload.view.*
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*
import kotlinx.android.synthetic.main.fragment_balance.view.progress_bar as progress_bar1
import kotlinx.android.synthetic.main.fragment_balance.view.tx_progress as tx_progress1

class BalanceFragment : Fragment() {

    private var viewProfile : View?=null
    val REQUEST_IMAGE_CAPTURE = 1
    lateinit var currentPhotoPath : String

    var pickImageFromAlbum=0
    var uriPhoto : Uri? = null
    var fbStorage : FirebaseStorage?=null
    var email: String = "default email"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        viewProfile = inflater.inflate(R.layout.fragment_balance, container, false)

        //카메라 촬영
        viewProfile!!.btn_camera.setOnClickListener(View.OnClickListener {
            //카메라 실행 코드
//            startCapture()
        })

        //동영상 선택
        viewProfile!!.btn_choose.setOnClickListener(View.OnClickListener {
            var photoPickerIntent = Intent(Intent.ACTION_PICK)
            photoPickerIntent.type = "video/*"
            startActivityForResult(photoPickerIntent, pickImageFromAlbum)
        })

        return viewProfile
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        //선택하기에 대한 실행
        if(requestCode == pickImageFromAlbum){
            uriPhoto = data?.data
            img_pre.visibility = View.VISIBLE
            img_pre.setVideoURI(uriPhoto)
            img_pre.setMediaController(MediaController(context))
            img_pre.start()
            btn_upload.visibility = View.VISIBLE

            btn_upload.setOnClickListener(View.OnClickListener {
                //progressBar 시작
                this.viewProfile!!.progress_bar.visibility = View.VISIBLE
                this.viewProfile!!.tx_progress.visibility = View.VISIBLE

                //layout 반투명화
                val paint = Paint()
                paint.alpha = 80
                this.viewProfile!!.balance_layout.setBackgroundColor(paint.getColor())
                this.viewProfile!!.img_pre.setBackgroundColor(paint.getColor())

                fbStorage = FirebaseStorage.getInstance()

                var email = (activity as MainPageActivity).email
                var timeStamp = SimpleDateFormat("yyMMdd_HH:mm").format(Date())
                var date = SimpleDateFormat("yyMMdd").format(Date())
                var imgFileName = "VIDEO_"+timeStamp+"_.mp4"
                var storageRef = fbStorage?.reference?.child(email)?.child("balance")?.child(date)?.child(imgFileName)

                storageRef?.putFile(uriPhoto!!)?.addOnSuccessListener {
                    Toast.makeText(context, "balance Video Uploaded", Toast.LENGTH_LONG).show()

                    //upload가 완료되면 balance page로 이동
                    changeFragment()
                }
            })
        }

        //사진촬영에 대한 실행
        if(requestCode == REQUEST_IMAGE_CAPTURE){

        }
    }


//    fun startCapture(){
//        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePicktureIntent ->
//            takePicktureIntent.resolveActivity(packageManager)?.also {
//                val photoFile: File? = try {
//                    createImageFile()
//                }
//                catch(ex: IOException){
//                    null
//                }
//                photoFile?.also {
//                    val photoURI : Uri = FileProvider.getUriForFile(
//                        this,
//                        "org.techtown.capturepicture.fileprovider",
//                        it
//                    )
//                    takePicktureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
//                    startActivityForResult(takePicktureIntent, REQUEST_IMAGE_CAPTURE)
//                }
//            }
//        }
//    }
//
//    fun createImageFile(): File {
//
//        val timeStamp : String = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
//        val storageDir : File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
//        return File.createTempFile(
//            "JPEG_${timeStamp}_",
//            ".jpg",
//            storageDir
//        ).apply{
//            currentPhotoPath = absolutePath
//        }
//    }

    //fragment  전환
    private fun changeFragment(){
        val activity = activity as MainPageActivity
        activity.setFragment(BalanceAnalyseFragment())
    }
}