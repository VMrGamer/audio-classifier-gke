variable "project_id" {
  type        = string
  description = "The ID of the GCP project in which the resource belongs"
}

variable "region" {
  type        = string
  default     = "us-central1"
  description = <<EOF
      The default region to manage resources in.
      If another region is specified on a regional resource, it will take precedence."
  EOF
}

variable "zone" {
  type        = string
  default     = "us-central1-a"
  description = <<EOF
      The default zone to manage resources in.
      Generally, this zone should be within the default region you specified.
      If another zone is specified on a zonal resource, it will take precedence.
  EOF
}

variable "location" {
  type        = string
  default     = "us-central1-a"
  description = <<EOF
      The location (region or zone) where the resources will be created in. \
      If you specify a region (such as us-west1), \
      the cluster will be a regional cluster with multiple masters spread across zones in the region, \
      and with default node locations in those zones as well"
  EOF
}

variable "credentials_file" {
  type        = string
  description = "The path to the credentials file of the service account used by terraform."
}

variable "cluster_name" {
  type        = string
  default     = "my-gke-cluster"
  description = "The name of the GKE cluster."
}

variable "node_pool_name" {
  type        = string
  default     = "my-node-pool"
  description = "The name of the GKE node pool."
}

variable "node_count" {
  type        = number
  default     = 1
  description = <<EOF
      The number of nodes per instance group. \
      This field can be used to update the number of nodes per instance group \
      but should not be used alongside autoscaling"
  EOF
}