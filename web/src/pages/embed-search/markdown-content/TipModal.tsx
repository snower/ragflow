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
            height: '75vh',
          },
        }}
        onOk={(e) => {
          setOpen(false);
          props?.onOk?.(e);
        }}
        onCancel={(e) => {
          setOpen(false);
          props?.onCancel?.(e);
        }}
      >
        {children}
      </Modal>
    </>
  );
};

export default TipModal;
