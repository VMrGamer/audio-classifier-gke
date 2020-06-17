terraform {
  backend "gcs" {
	credentials	= "C:\\Users\\V3Anom\\Desktop\\MP\\audioclassification-279617-7500baefae3e.json"
    bucket      		= "audioclassification-279617-bucket"
    prefix     			= "terraform/state"
  }
}