# autoware_ground_filter

## Purpose

The `autoware_ground_filter` is a node that remove the ground points from the input pointcloud.

## Inner-workings / Algorithms

Detail description of each ground segmentation algorithm is in the following links.

| Filter Name   | Description                                                                                                | Detail                        |
| ------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------- |
| ground_filter | A method of removing the ground based on the geometrical relationship between points lined up on radiation | [link](docs/ground-filter.md) |

## Inputs / Outputs

### Input

| Name              | Type                            | Description       |
| ----------------- | ------------------------------- | ----------------- |
| `~/input/points`  | `sensor_msgs::msg::PointCloud2` | reference points  |
| `~/input/indices` | `pcl_msgs::msg::Indices`        | reference indices |

### Output

| Name              | Type                            | Description     |
| ----------------- | ------------------------------- | --------------- |
| `~/output/points` | `sensor_msgs::msg::PointCloud2` | filtered points |

## Parameters

### Node Parameters

| Name                 | Type   | Default Value | Description                           |
| -------------------- | ------ | ------------- | ------------------------------------- |
| `input_frame`        | string | " "           | input frame id                        |
| `output_frame`       | string | " "           | output frame id                       |
| `has_static_tf_only` | bool   | false         | flag to listen TF only once           |
| `max_queue_size`     | int    | 5             | max queue size of input/output topics |
| `use_indices`        | bool   | false         | flag to use pointcloud indices        |
| `latched_indices`    | bool   | false         | flag to latch pointcloud indices      |
| `approximate_sync`   | bool   | false         | flag to use approximate sync option   |

## Assumptions / Known limits

Implemented based on pcl_perception [1] because of [this issue](https://github.com/ros-perception/perception_pcl/issues/9).

## References/External links

[1] <https://github.com/ros-perception/perception_pcl/blob/ros2/pcl_ros/src/pcl_ros/filters/filter.cpp>
