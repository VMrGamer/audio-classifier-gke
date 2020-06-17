variable "project_id" {
  type        = string
  description = "The name of the GCP project."
}

variable "service_account_name" {
  type        = string
  description = "The name of the service account used by terraform."
}

variable "credentials_file" {
  type        = string
  description = "The path to the credentials file of the service account used by terraform."
}

variable "bucket_name" {
  type        = string
  description = "The name of the terraform backend bucket."
}

variable "location" {
  type        = string
  default     = "US"
  description = "The GCS location."
}