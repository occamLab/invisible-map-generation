# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: map_data.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='map_data.proto',
  package='',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x0emap_data.proto\"*\n\x07PGTrans\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\"3\n\x05PGRot\x12\t\n\x01x\x18\x01 \x01(\x02\x12\t\n\x01y\x18\x02 \x01(\x02\x12\t\n\x01z\x18\x03 \x01(\x02\x12\t\n\x01w\x18\x04 \x01(\x02\"\"\n\rNeighborsList\x12\x11\n\tneighbors\x18\x01 \x03(\x05\"M\n\x06PGTagV\x12\x1d\n\x0btranslation\x18\x01 \x01(\x0b\x32\x08.PGTrans\x12\x18\n\x08rotation\x18\x02 \x01(\x0b\x32\x06.PGRot\x12\n\n\x02id\x18\x03 \x01(\x05\"\x83\x01\n\x08PGCloudV\x12\x1d\n\x0btranslation\x18\x01 \x01(\x0b\x32\x08.PGTrans\x12\x18\n\x08rotation\x18\x02 \x01(\x0b\x32\x06.PGRot\x12\x15\n\x08\x63loud_id\x18\x03 \x01(\tH\x00\x88\x01\x01\x12\x11\n\x04name\x18\x04 \x01(\tH\x01\x88\x01\x01\x42\x0b\n\t_cloud_idB\x07\n\x05_name\"\x87\x02\n\x07PGOdomV\x12\x1d\n\x0btranslation\x18\x01 \x01(\x0b\x32\x08.PGTrans\x12\x18\n\x08rotation\x18\x02 \x01(\x0b\x32\x06.PGRot\x12\x0e\n\x06poseId\x18\x03 \x01(\x05\x12\x14\n\x07\x61\x64jChi2\x18\x04 \x01(\x02H\x00\x88\x01\x01\x12\x16\n\tcloudChi2\x18\x05 \x01(\x02H\x01\x88\x01\x01\x12\x14\n\x07vizTags\x18\x06 \x01(\x05H\x02\x88\x01\x01\x12\x15\n\x08vizCloud\x18\x07 \x01(\x05H\x03\x88\x01\x01\x12%\n\rall_neighbors\x18\x08 \x03(\x0b\x32\x0e.NeighborsListB\n\n\x08_adjChi2B\x0c\n\n_cloudChi2B\n\n\x08_vizTagsB\x0b\n\t_vizCloud\"R\n\x0bPGWaypointV\x12\x1d\n\x0btranslation\x18\x01 \x01(\x0b\x32\x08.PGTrans\x12\x18\n\x08rotation\x18\x02 \x01(\x0b\x32\x06.PGRot\x12\n\n\x02id\x18\x03 \x01(\t\"\x99\x01\n\x06PGData\x12\x1d\n\x0ctag_vertices\x18\x01 \x03(\x0b\x32\x07.PGTagV\x12!\n\x0e\x63loud_vertices\x18\x02 \x03(\x0b\x32\t.PGCloudV\x12#\n\x11odometry_vertices\x18\x03 \x03(\x0b\x32\x08.PGOdomV\x12(\n\x12waypoints_vertices\x18\x04 \x03(\x0b\x32\x0c.PGWaypointVb\x06proto3'
)




_PGTRANS = _descriptor.Descriptor(
  name='PGTrans',
  full_name='PGTrans',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='PGTrans.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='y', full_name='PGTrans.y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='z', full_name='PGTrans.z', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=18,
  serialized_end=60,
)


_PGROT = _descriptor.Descriptor(
  name='PGRot',
  full_name='PGRot',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='x', full_name='PGRot.x', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='y', full_name='PGRot.y', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='z', full_name='PGRot.z', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='w', full_name='PGRot.w', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=62,
  serialized_end=113,
)


_NEIGHBORSLIST = _descriptor.Descriptor(
  name='NeighborsList',
  full_name='NeighborsList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='neighbors', full_name='NeighborsList.neighbors', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=115,
  serialized_end=149,
)


_PGTAGV = _descriptor.Descriptor(
  name='PGTagV',
  full_name='PGTagV',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='translation', full_name='PGTagV.translation', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='rotation', full_name='PGTagV.rotation', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='id', full_name='PGTagV.id', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=151,
  serialized_end=228,
)


_PGCLOUDV = _descriptor.Descriptor(
  name='PGCloudV',
  full_name='PGCloudV',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='translation', full_name='PGCloudV.translation', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='rotation', full_name='PGCloudV.rotation', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cloud_id', full_name='PGCloudV.cloud_id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='name', full_name='PGCloudV.name', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='_cloud_id', full_name='PGCloudV._cloud_id',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
    _descriptor.OneofDescriptor(
      name='_name', full_name='PGCloudV._name',
      index=1, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=231,
  serialized_end=362,
)


_PGODOMV = _descriptor.Descriptor(
  name='PGOdomV',
  full_name='PGOdomV',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='translation', full_name='PGOdomV.translation', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='rotation', full_name='PGOdomV.rotation', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='poseId', full_name='PGOdomV.poseId', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='adjChi2', full_name='PGOdomV.adjChi2', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cloudChi2', full_name='PGOdomV.cloudChi2', index=4,
      number=5, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vizTags', full_name='PGOdomV.vizTags', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='vizCloud', full_name='PGOdomV.vizCloud', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='all_neighbors', full_name='PGOdomV.all_neighbors', index=7,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='_adjChi2', full_name='PGOdomV._adjChi2',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
    _descriptor.OneofDescriptor(
      name='_cloudChi2', full_name='PGOdomV._cloudChi2',
      index=1, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
    _descriptor.OneofDescriptor(
      name='_vizTags', full_name='PGOdomV._vizTags',
      index=2, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
    _descriptor.OneofDescriptor(
      name='_vizCloud', full_name='PGOdomV._vizCloud',
      index=3, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=365,
  serialized_end=628,
)


_PGWAYPOINTV = _descriptor.Descriptor(
  name='PGWaypointV',
  full_name='PGWaypointV',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='translation', full_name='PGWaypointV.translation', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='rotation', full_name='PGWaypointV.rotation', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='id', full_name='PGWaypointV.id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=630,
  serialized_end=712,
)


_PGDATA = _descriptor.Descriptor(
  name='PGData',
  full_name='PGData',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='tag_vertices', full_name='PGData.tag_vertices', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cloud_vertices', full_name='PGData.cloud_vertices', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='odometry_vertices', full_name='PGData.odometry_vertices', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='waypoints_vertices', full_name='PGData.waypoints_vertices', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=715,
  serialized_end=868,
)

_PGTAGV.fields_by_name['translation'].message_type = _PGTRANS
_PGTAGV.fields_by_name['rotation'].message_type = _PGROT
_PGCLOUDV.fields_by_name['translation'].message_type = _PGTRANS
_PGCLOUDV.fields_by_name['rotation'].message_type = _PGROT
_PGCLOUDV.oneofs_by_name['_cloud_id'].fields.append(
  _PGCLOUDV.fields_by_name['cloud_id'])
_PGCLOUDV.fields_by_name['cloud_id'].containing_oneof = _PGCLOUDV.oneofs_by_name['_cloud_id']
_PGCLOUDV.oneofs_by_name['_name'].fields.append(
  _PGCLOUDV.fields_by_name['name'])
_PGCLOUDV.fields_by_name['name'].containing_oneof = _PGCLOUDV.oneofs_by_name['_name']
_PGODOMV.fields_by_name['translation'].message_type = _PGTRANS
_PGODOMV.fields_by_name['rotation'].message_type = _PGROT
_PGODOMV.fields_by_name['all_neighbors'].message_type = _NEIGHBORSLIST
_PGODOMV.oneofs_by_name['_adjChi2'].fields.append(
  _PGODOMV.fields_by_name['adjChi2'])
_PGODOMV.fields_by_name['adjChi2'].containing_oneof = _PGODOMV.oneofs_by_name['_adjChi2']
_PGODOMV.oneofs_by_name['_cloudChi2'].fields.append(
  _PGODOMV.fields_by_name['cloudChi2'])
_PGODOMV.fields_by_name['cloudChi2'].containing_oneof = _PGODOMV.oneofs_by_name['_cloudChi2']
_PGODOMV.oneofs_by_name['_vizTags'].fields.append(
  _PGODOMV.fields_by_name['vizTags'])
_PGODOMV.fields_by_name['vizTags'].containing_oneof = _PGODOMV.oneofs_by_name['_vizTags']
_PGODOMV.oneofs_by_name['_vizCloud'].fields.append(
  _PGODOMV.fields_by_name['vizCloud'])
_PGODOMV.fields_by_name['vizCloud'].containing_oneof = _PGODOMV.oneofs_by_name['_vizCloud']
_PGWAYPOINTV.fields_by_name['translation'].message_type = _PGTRANS
_PGWAYPOINTV.fields_by_name['rotation'].message_type = _PGROT
_PGDATA.fields_by_name['tag_vertices'].message_type = _PGTAGV
_PGDATA.fields_by_name['cloud_vertices'].message_type = _PGCLOUDV
_PGDATA.fields_by_name['odometry_vertices'].message_type = _PGODOMV
_PGDATA.fields_by_name['waypoints_vertices'].message_type = _PGWAYPOINTV
DESCRIPTOR.message_types_by_name['PGTrans'] = _PGTRANS
DESCRIPTOR.message_types_by_name['PGRot'] = _PGROT
DESCRIPTOR.message_types_by_name['NeighborsList'] = _NEIGHBORSLIST
DESCRIPTOR.message_types_by_name['PGTagV'] = _PGTAGV
DESCRIPTOR.message_types_by_name['PGCloudV'] = _PGCLOUDV
DESCRIPTOR.message_types_by_name['PGOdomV'] = _PGODOMV
DESCRIPTOR.message_types_by_name['PGWaypointV'] = _PGWAYPOINTV
DESCRIPTOR.message_types_by_name['PGData'] = _PGDATA
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

PGTrans = _reflection.GeneratedProtocolMessageType('PGTrans', (_message.Message,), {
  'DESCRIPTOR' : _PGTRANS,
  '__module__' : 'map_data_pb2'
  # @@protoc_insertion_point(class_scope:PGTrans)
  })
_sym_db.RegisterMessage(PGTrans)

PGRot = _reflection.GeneratedProtocolMessageType('PGRot', (_message.Message,), {
  'DESCRIPTOR' : _PGROT,
  '__module__' : 'map_data_pb2'
  # @@protoc_insertion_point(class_scope:PGRot)
  })
_sym_db.RegisterMessage(PGRot)

NeighborsList = _reflection.GeneratedProtocolMessageType('NeighborsList', (_message.Message,), {
  'DESCRIPTOR' : _NEIGHBORSLIST,
  '__module__' : 'map_data_pb2'
  # @@protoc_insertion_point(class_scope:NeighborsList)
  })
_sym_db.RegisterMessage(NeighborsList)

PGTagV = _reflection.GeneratedProtocolMessageType('PGTagV', (_message.Message,), {
  'DESCRIPTOR' : _PGTAGV,
  '__module__' : 'map_data_pb2'
  # @@protoc_insertion_point(class_scope:PGTagV)
  })
_sym_db.RegisterMessage(PGTagV)

PGCloudV = _reflection.GeneratedProtocolMessageType('PGCloudV', (_message.Message,), {
  'DESCRIPTOR' : _PGCLOUDV,
  '__module__' : 'map_data_pb2'
  # @@protoc_insertion_point(class_scope:PGCloudV)
  })
_sym_db.RegisterMessage(PGCloudV)

PGOdomV = _reflection.GeneratedProtocolMessageType('PGOdomV', (_message.Message,), {
  'DESCRIPTOR' : _PGODOMV,
  '__module__' : 'map_data_pb2'
  # @@protoc_insertion_point(class_scope:PGOdomV)
  })
_sym_db.RegisterMessage(PGOdomV)

PGWaypointV = _reflection.GeneratedProtocolMessageType('PGWaypointV', (_message.Message,), {
  'DESCRIPTOR' : _PGWAYPOINTV,
  '__module__' : 'map_data_pb2'
  # @@protoc_insertion_point(class_scope:PGWaypointV)
  })
_sym_db.RegisterMessage(PGWaypointV)

PGData = _reflection.GeneratedProtocolMessageType('PGData', (_message.Message,), {
  'DESCRIPTOR' : _PGDATA,
  '__module__' : 'map_data_pb2'
  # @@protoc_insertion_point(class_scope:PGData)
  })
_sym_db.RegisterMessage(PGData)


# @@protoc_insertion_point(module_scope)
