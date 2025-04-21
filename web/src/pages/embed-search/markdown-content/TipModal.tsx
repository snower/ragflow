import { InfoCircleOutlined } from '@ant-design/icons';
import { Modal, ModalProps } from 'antd';
import React, { PropsWithChildren, useState } from 'react';

type Props = Omit<ModalProps, 'open'> & {};
const TipModal: React.FC<PropsWithChildren<Props>> = ({
  children,
  ...props
}) => {
  const [open, setOpen] = useState(false);
  return (
    <>
      <InfoCircleOutlined
        onClick={() => {
          setOpen(true);
        }}
        style={{
          padding: '0 6px',
        }}
      />
      <Modal
        open={open}
        {...props}
        styles={{
          body: {
            overflow: 'hidden',
          },
        }}
        width="960px"
        className="tipModal"
        onOk={(e) => {
          setOpen(false);
          props?.onOk?.(e);
        }}
        onCancel={(e) => {
          setOpen(false);
          props?.onCancel?.(e);
        }}
        footer={null}
      >
        {children}
      </Modal>
    </>
  );
};

export default TipModal;
