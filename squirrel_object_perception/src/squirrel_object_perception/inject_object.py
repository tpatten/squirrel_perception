#!/usr/bin/env python
#
# Simple node to inject an object into the database
# For testing purposes.
#
# Jan 2016, Michael Zillich <michael.zillich@tuwien.ac.at>

import getopt, sys
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from squirrel_object_perception_msgs.msg import SceneObject, BCylinder
from squirrel_planning_knowledge_msgs.srv import AddObjectService, \
    AddObjectServiceRequest, UpdateObjectService, UpdateObjectServiceRequest

def add_object_to_db(id, category, pose, size):
    add_object = rospy.ServiceProxy('/kcl_rosplan/add_object', AddObjectService)
    try:
        rospy.wait_for_service('/kcl_rosplan/add_object', timeout=3)
        obj = SceneObject()
        obj.id = id
        obj.category = category
        obj.pose = pose
        obj.bounding_cylinder = BCylinder()
        obj.bounding_cylinder.diameter = size
        obj.bounding_cylinder.height = size
        obj.header.frame_id = "map"
        # note: we leave the cloud empty: obj.cloud
        request = AddObjectServiceRequest()
        request.object = obj
        resp = add_object(request)
    except rospy.ServiceException as exc:
        print("Service did not process request: " + str(exc))
        resp = False
    return resp

def usage():
    print "Injects an object with ID, category/class, size and 2D position (map frame) into the scene database."
    print "-i id -c category -s size -x xpos -y ypos"

if __name__ == '__main__':

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:c:s:x:y:", ["help", "id=", "category=", "size=", "xpos=", "ypos="])

        id = ""
        category = ""
        pose = Pose()
        pose.position.x = 0.0
        pose.position.y = 0.0
        pose.position.z = 0.0
        pose.orientation.x = pose.orientation.y = pose.orientation.z = 0.0
        pose.orientation.w = 1.0
        size = 0.0

        for o, a in opts:
            if o == "-i":
                id = a
            elif o == "-c":
                category = a
            elif o == "-s":
                size = float(a)
            elif o == "-x":
                print "-x"
                print a
                pose.position.x = float(a)
            elif o == "-y":
                print "-y"
                print a
                pose.position.y = float(a)
            elif o in ("-h", "--help"):
                usage()
                sys.exit()
            else:
                assert False, "unhandled option"
        print "injecting object '" + str(id) + "' of category '" + category + "' with size " + str(size) \
         + " at position [" + str(pose.position.x) + " " + str(pose.position.y) + "]"
        add_object_to_db(id, category, pose, size)

    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
    except rospy.ROSInterruptException:
        pass
