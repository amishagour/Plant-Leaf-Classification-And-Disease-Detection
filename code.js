<script>
function readURL(input) {
if (input.files && input.files[0]) {
var reader = new FileReader();

reader.onload = function (e) {
var blaho = document.getElementById('blah');
blaho.src = e.target.result;
};

reader.readAsDataURL(input.files[0]);

var pred = document.getElementById('predict')
pred.innerText = 'The result is :'
  }
}
</script>
